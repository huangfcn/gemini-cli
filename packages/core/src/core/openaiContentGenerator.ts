/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Content as _Content,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
  Part,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
// import stack from 'mnemonist/stack';

interface OpenAIMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface OpenAIConfig {
  endpoint: string;
  model: string;
  apiKey?: string;
  debugOpenai?: boolean;
}

/**
 * A ContentGenerator implementation that uses OpenAI-compatible APIs.
 * This works with OpenAI, Azure OpenAI, Ollama, and other compatible endpoints.
 */
export class OpenAICompatibleContentGenerator implements ContentGenerator {
  private readonly endpoint: string;
  private readonly model: string;
  private readonly apiKey?: string;
  private readonly debugOpenai: boolean;

  constructor(config: OpenAIConfig) {
    this.endpoint = config.endpoint;
    this.model = config.model;
    this.apiKey = config.apiKey;
    this.debugOpenai = config.debugOpenai ?? false;
  }

  /**
   * Converts Gemini-style contents to OpenAI message format.
   */
  private convertContentsToMessages(
    contents: GenerateContentParameters['contents'],
  ): OpenAIMessage[] {
    const messages: OpenAIMessage[] = [];

    if (!contents) {
      return messages;
    }

    // Handle string input
    if (typeof contents === 'string') {
      messages.push({ role: 'user', content: contents });
      return messages;
    }

    // Handle array input
    if (Array.isArray(contents)) {
      for (const item of contents) {
        // Check if it's a Content object (has role and parts)
        if (typeof item === 'object' && item !== null && 'role' in item) {
          const content = item;
          const role = content.role === 'model' ? 'assistant' : 'user';
          const text = this.extractTextFromParts(content.parts ?? []);
          if (text) {
            messages.push({ role, content: text });
          }
        } else if (typeof item === 'string') {
          // It's a string PartUnion
          messages.push({ role: 'user', content: item });
        } else if (typeof item === 'object' && item !== null) {
          // It's a Part object
          const part = item as Part;
          const text = part.text ?? '';
          if (text) {
            messages.push({ role: 'user', content: text });
          }
        }
      }
      return messages;
    }

    // Handle single Content object
    if (typeof contents === 'object' && 'role' in contents) {
      const content = contents;
      const role = content.role === 'model' ? 'assistant' : 'user';
      const text = this.extractTextFromParts(content.parts ?? []);
      if (text) {
        messages.push({ role, content: text });
      }
      return messages;
    }

    // Handle single Part object
    if (typeof contents === 'object' && 'text' in contents) {
      const part = contents;
      if (part.text) {
        messages.push({ role: 'user', content: part.text });
      }
    }

    return messages;
  }

  /**
   * Extracts text from an array of Part objects.
   */
  private extractTextFromParts(parts: Part[]): string {
    return parts
      .map((part) => part.text ?? '')
      .filter((text) => text.length > 0)
      .join('');
  }

  /**
   * Creates the HTTP headers for API requests.
   */
  private createHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  /**
   * Creates a GenerateContentResponse from OpenAI response data.
   */
  private createResponse(responseText: string): GenerateContentResponse {
    return {
      candidates: [
        {
          index: 0,
          content: {
            role: 'model',
            parts: [{ text: responseText }],
          },
          finishReason: 'STOP',
          safetyRatings: [],
        },
      ],
      text: () => responseText,
      functionCall: undefined,
      functionCalls: undefined,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any;
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = this.convertContentsToMessages(request.contents);

    try {
      if (this.debugOpenai) {
        const fs = await import('node:fs');
        const path = await import('node:path');
        const process = await import('node:process');
        const logPath = path.join(process.cwd(), 'debug_openai.log');
        const headers = this.createHeaders();
        const lastMessage =
          messages.length > 0 ? messages[messages.length - 1] : 'No messages';
        // Log stack trace to identify caller
        // const stack = new Error().stack;
        const logMsg = `[${new Date().toISOString()}] === Request to LLM ===
Endpoint: ${this.endpoint}
Method: POST
Model: ${this.model}
Headers: ${JSON.stringify(headers)}
Last Message: ${JSON.stringify(lastMessage)}
Full Message Count: ${messages.length}
\n`;
        fs.appendFileSync(logPath, logMsg);
      }
    } catch (_e) {
      // ignore logging errors
    }

    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: this.createHeaders(),
      body: JSON.stringify({
        model: this.model,
        messages,
        temperature: request.config?.temperature,
        max_tokens: request.config?.maxOutputTokens,
      }),
      signal: request.config?.abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      try {
        if (this.debugOpenai) {
          const fs = await import('node:fs');
          const path = await import('node:path');
          const process = await import('node:process');
          const logPath = path.join(process.cwd(), 'debug_openai.log');
          fs.appendFileSync(
            logPath,
            `[${new Date().toISOString()}] Error (${response.status}): ${errorText}\n\n`,
          );
        }
      } catch (_e) {
        // ignore
      }
      throw new Error(
        `OpenAI compatible API request failed with status ${response.status}: ${errorText}`,
      );
    }

    const data = await response.json();
    const responseText = data.choices?.[0]?.message?.content ?? '';

    try {
      if (this.debugOpenai) {
        const fs = await import('node:fs');
        const path = await import('node:path');
        const process = await import('node:process');
        const logPath = path.join(process.cwd(), 'debug_openai.log');
        fs.appendFileSync(
          logPath,
          `[${new Date().toISOString()}] === Response from LLM ===\n${JSON.stringify(data)}\n\n`,
        );
      }
    } catch (_e) {
      // ignore
    }

    return this.createResponse(responseText);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.convertContentsToMessages(request.contents);

    try {
      if (this.debugOpenai) {
        const fs = await import('node:fs');
        const path = await import('node:path');
        const process = await import('node:process');
        const logPath = path.join(process.cwd(), 'debug_openai.log');
        fs.appendFileSync(
          logPath,
          `[${new Date().toISOString()}] === Request to LLM (Stream) ===
Endpoint: ${this.endpoint}
Method: POST
Model: ${this.model}
Last Message: ${JSON.stringify(messages[messages.length - 1])}
\n`,
        );
      }
    } catch (_e) {
      // ignore
    }

    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: this.createHeaders(),
      body: JSON.stringify({
        model: this.model,
        messages,
        temperature: request.config?.temperature,
        max_tokens: request.config?.maxOutputTokens,
        stream: true,
      }),
      signal: request.config?.abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenAI compatible API request failed with status ${response.status}: ${errorText}`,
      );
    }

    const reader = response.body?.getReader();
    if (!reader) {
      // Fallback to non-streaming if no response body
      const generateContent = this.generateContent.bind(this);
      return (async function* () {
        try {
          const result = await generateContent(request);
          yield result;
        } catch (_e) {
          throw new Error(
            'Response body is not a readable stream and fallback failed: ' + _e,
          );
        }
      })();
    }

    // Capture for use in generator closure
    const streamReader: ReadableStreamDefaultReader<Uint8Array> = reader;
    const decoder = new TextDecoder();
    const { debugOpenai } = this;
    const createResponse = this.createResponse.bind(this);

    async function* generator(): AsyncGenerator<GenerateContentResponse> {
      let buffer = '';
      let fullResponse = ''; // Accumulate full response for logging

      while (true) {
        const { done, value } = await streamReader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6).trim();
            if (dataStr === '[DONE]') {
              // Log the accumulated response when stream completes
              try {
                if (debugOpenai) {
                  const fs = await import('node:fs');
                  const path = await import('node:path');
                  const process = await import('node:process');
                  const logPath = path.join(process.cwd(), 'debug_openai.log');
                  fs.appendFileSync(
                    logPath,
                    `[${new Date().toISOString()}] === Response from LLM (Stream Complete) ===\n${fullResponse}\n\n`,
                  );
                }
              } catch (_e) {
                // ignore logging errors
              }
              return;
            }
            try {
              const data = JSON.parse(dataStr);
              const responseText = data.choices?.[0]?.delta?.content ?? '';
              if (responseText) {
                fullResponse += responseText; // Accumulate for logging
                yield createResponse(responseText);
              }
            } catch {
              // Malformed JSON, skip this chunk
            }
          }
        }
      }

      // Log if stream ends without [DONE] marker
      try {
        if (debugOpenai && fullResponse) {
          const fs = await import('node:fs');
          const path = await import('node:path');
          const process = await import('node:process');
          const logPath = path.join(process.cwd(), 'debug_openai.log');
          fs.appendFileSync(
            logPath,
            `[${new Date().toISOString()}] === Response from LLM (Stream End) ===\n${fullResponse}\n\n`,
          );
        }
      } catch (_e) {
        // ignore logging errors
      }
    }

    return generator();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Simple token approximation: ~4 characters per token
    let text = '';

    if (typeof request.contents === 'string') {
      text = request.contents;
    } else if (Array.isArray(request.contents)) {
      for (const item of request.contents) {
        if (typeof item === 'string') {
          text += item;
        } else if (typeof item === 'object' && item !== null) {
          if ('role' in item) {
            // Content object
            const content = item;
            text += this.extractTextFromParts(content.parts ?? []);
          } else if ('text' in item) {
            // Part object
            const part = item;
            text += part.text ?? '';
          }
        }
      }
    } else if (
      typeof request.contents === 'object' &&
      request.contents !== null
    ) {
      if ('role' in request.contents) {
        const content = request.contents;
        text = this.extractTextFromParts(content.parts ?? []);
      } else if ('text' in request.contents) {
        const part = request.contents;
        text = part.text ?? '';
      }
    }

    return {
      totalTokens: Math.ceil(text.length / 4),
    };
  }

  embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'embedContent is not implemented for OpenAI-compatible generators.',
    );
  }
}
