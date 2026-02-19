import 'dotenv/config';

export interface AIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface AIResponse {
  content: string;
  tokens?: number;
  provider?: string;
  finishReason?: string;
}

export interface AIConfig {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  stream?: boolean;
}

export class AIService {
  private providers: string[];

  constructor() {
    this.providers = this.detectAvailableProviders();
    console.log(`🤖 Available AI providers: ${this.providers.join(', ')}`);
  }

  private detectAvailableProviders(): string[] {
    const available: string[] = [];

    if (process.env.OPENROUTER_API_KEY) {
      available.push('openrouter');
    }

    available.push('fallback');

    const preferredProvider = process.env.AI_PROVIDER || 'openrouter';
    if (available.includes(preferredProvider)) {
      const index = available.indexOf(preferredProvider);
      if (index > 0) {
        available.splice(index, 1);
        available.unshift(preferredProvider);
      }
    }

    return available;
  }

  async chat(messages: AIMessage[], config: AIConfig = {}): Promise<AIResponse> {
    return this.tryWithProviders(messages, config);
  }

  private async tryWithProviders(messages: AIMessage[], config: AIConfig, attempt = 0): Promise<AIResponse> {
    if (attempt >= this.providers.length) {
      throw new Error('All AI providers failed');
    }

    const provider = this.providers[attempt];

    try {
      switch (provider) {
        case 'openrouter':
          return await this.callOpenRouterAPI(messages, config);
        case 'fallback':
          return this.fallbackResponse(messages);
        default:
          throw new Error(`Unknown provider: ${provider}`);
      }
    } catch (error: any) {
      console.warn(`⚠️ Provider ${provider} failed:`, error?.message || error);

      const emsg = String(error?.message || error || '');
      if (emsg.includes('region') || emsg.includes('not available')) {
        console.log(`🌍 ${provider} unavailable in your region, trying next...`);
      }

      return this.tryWithProviders(messages, config, attempt + 1);
    }
  }

  // --- OpenRouter via fetch (DeepSeek) ---
  private async callOpenRouterAPI(messages: AIMessage[], config: AIConfig): Promise<AIResponse> {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      throw new Error('OpenRouter API key not configured');
    }

    const model = config.model || process.env.OPENROUTER_MODEL || 'deepseek/deepseek-r1-0528:free';
    const maxTokens = config.maxTokens || parseInt(process.env.DEFAULT_MAX_TOKENS || '512');
    const temperature = typeof config.temperature === 'number'
      ? config.temperature
      : parseFloat(process.env.DEFAULT_TEMPERATURE || '0.7');

    const aiMessages = this.limitContext(messages);

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': process.env.APP_URL || 'http://localhost',
        'X-Title': process.env.APP_NAME || 'AI Assistant'
      },
      body: JSON.stringify({
        model,
        messages: aiMessages,
        temperature,
        max_tokens: maxTokens,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`OpenRouter HTTP ${response.status}: ${text}`);
    }

    const data: any = await response.json();

    return {
      content: data.choices?.[0]?.message?.content || '',
      tokens: data.usage?.total_tokens,
      provider: 'openrouter',
      finishReason: data.choices?.[0]?.finish_reason,
    };
  }

  private fallbackResponse(messages: AIMessage[]): AIResponse {
    const lastMessage = messages[messages.length - 1]?.content || '';

    const responses = [
      'Привет! Я ваш AI ассистент. Сейчас работаю в демо-режиме. Добавьте OPENROUTER_API_KEY в .env для полноценной работы.',
      'Чтобы получить реальные ответы от AI, настройте OpenRouter и добавьте OPENROUTER_API_KEY в .env.',
      'Демо-режим. Получите API ключ на OpenRouter и добавьте OPENROUTER_API_KEY в .env'
    ];

    const response = responses[Math.floor(Math.random() * responses.length)];

    return {
      content: `${response}\n\nВаш запрос: "${lastMessage.substring(0, 100)}..."`,
      provider: 'fallback'
    };
  }

  async generateTitle(firstMessage: string): Promise<string> {
    try {
      const response = await this.chat([
        {
          role: 'system',
          content: 'Сгенерируй краткий заголовок (максимум 5 слов) для этого сообщения. Возвращай только заголовок, без кавычек. Используй тот же язык, что и в сообщении.'
        },
        {
          role: 'user',
          content: firstMessage
        }
      ], {
        maxTokens: 50,
        temperature: 0.3
      });

      return response.content.trim() || 'New Chat';
    } catch (error) {
      console.warn('Failed to generate title:', error);
      return firstMessage.substring(0, 50) + '...';
    }
  }

  private limitContext(messages: AIMessage[]): AIMessage[] {
    const maxTokens = parseInt(process.env.MAX_CONTEXT_TOKENS || '4000');
    let totalTokens = 0;
    const limitedMessages: AIMessage[] = [];

    const systemMessage = messages.find(m => m.role === 'system');
    if (systemMessage) {
      const estimatedTokens = Math.ceil(systemMessage.content.length / 4);
      totalTokens += estimatedTokens;
      limitedMessages.push(systemMessage);
    }

    const otherMessages = messages.filter(m => m.role !== 'system').reverse();

    for (const message of otherMessages) {
      const estimatedTokens = Math.ceil(message.content.length / 4);

      if (totalTokens + estimatedTokens > maxTokens) {
        break;
      }

      totalTokens += estimatedTokens;
      limitedMessages.push(message);
    }

    limitedMessages.sort((a, b) => {
      if (a.role === 'system') return -1;
      if (b.role === 'system') return 1;
      return 0;
    });

    return limitedMessages;
  }

  async testConnection(): Promise<{ success: boolean; provider?: string }> {
    try {
      const result = await this.chat([
        { role: 'user', content: 'Hello' }
      ], { maxTokens: 5 });

      return {
        success: true,
        provider: result.provider || 'unknown'
      };
    } catch (error: any) {
      console.error('AI connection test failed:', error?.message || error);
      return { success: false };
    }
  }

  // --- Streaming via SSE ---
  async *streamChat(messages: AIMessage[], config: AIConfig = {}): AsyncGenerator<string> {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      throw new Error('OpenRouter API key not configured for streaming');
    }

    const model = config.model || process.env.OPENROUTER_MODEL || 'deepseek/deepseek-r1-0528:free';

    const aiMessages = this.limitContext(messages);

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': process.env.APP_URL || 'http://localhost',
        'X-Title': process.env.APP_NAME || 'AI Assistant'
      },
      body: JSON.stringify({
        model,
        messages: aiMessages,
        stream: true,
      }),
    });

    if (!response.body) {
      throw new Error('No response body for streaming');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n').filter(line => line.startsWith('data: '));

      for (const line of lines) {
        const json = line.replace('data: ', '').trim();
        if (json === '[DONE]') return;

        try {
          const parsed = JSON.parse(json);
          const content = parsed.choices?.[0]?.delta?.content;
          if (content) {
            yield content;
          }
        } catch (err) {
          console.warn('Failed to parse stream chunk:', err);
        }
      }
    }
  }
}

export const aiService = new AIService();
