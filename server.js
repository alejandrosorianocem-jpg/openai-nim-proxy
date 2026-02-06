// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it (QwQ, R1, etc.)
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 PREFILL TOGGLE - Forces models to think step-by-step
const ENABLE_PREFILL = true; // Set to false to disable automatic reasoning prefill

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'glm-4': 'zhipuai/glm-4-9b-chat',
  'glm-7': 'meta/llama-3.1-70b-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    prefill_enabled: ENABLE_PREFILL
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    console.log('📥 Incoming request:', { model, messageCount: messages?.length, stream });
    
    // Validate API key
    if (!NIM_API_KEY) {
      console.error('❌ NIM_API_KEY not set');
      return res.status(500).json({
        error: { message: 'NIM_API_KEY environment variable not set', type: 'configuration_error', code: 500 }
      });
    }
    
    // Validate messages
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      console.error('❌ Invalid messages format');
      return res.status(400).json({
        error: { message: 'Messages must be a non-empty array', type: 'invalid_request_error', code: 400 }
      });
    }
    
    // Try model in this order:
    // 1. Check if it's in mapping (for convenience aliases)
    // 2. Try the model name directly
    // 3. Only fallback if both fail
    
    let nimModel = MODEL_MAPPING[model];
    
    if (nimModel) {
      console.log('✅ Using mapped model:', model, '→', nimModel);
    } else {
      console.log('🔍 Model not in mapping, trying direct:', model);
      nimModel = model; // Use directly first
    }
    
    console.log('🔄 Using model:', nimModel);
    
    // Apply prefill if enabled
    let processedMessages = [...messages];
    if (ENABLE_PREFILL) {
      // Add system message if not present
      if (processedMessages[0]?.role !== 'system') {
        processedMessages.unshift({
          role: 'system',
          content: 'You are a helpful AI assistant. Always think through problems step-by-step before answering. Show your reasoning when relevant.'
        });
      }
      
      // Add assistant prefill to trigger reasoning
      processedMessages.push({
        role: 'assistant',
        content: 'Let me think through this carefully:\n\n'
      });
      
      console.log('✨ Prefill applied');
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature !== undefined ? temperature : 0.7,
      top_p: 0.9,
      max_tokens: max_tokens || 1024,
      stream: stream || false
    };
    
    // Only add extra_body if thinking mode is enabled (for specific models)
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    console.log('📤 Sending to NIM:', { model: nimRequest.model, max_tokens: nimRequest.max_tokens });
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    console.log('✅ NIM response received');
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('❌ Proxy error:', error.message);
    console.error('📋 Error details:', {
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      model: error.config?.data ? JSON.parse(error.config.data).model : 'unknown'
    });
    
    // Enhanced error response
    const errorMessage = error.response?.data?.detail || 
                        error.response?.data?.error?.message || 
                        error.message || 
                        'Internal server error';
    
    res.status(error.response?.status || 500).json({
      error: {
        message: `NIM API Error: ${errorMessage}`,
        type: 'invalid_request_error',
        code: error.response?.status || 500,
        details: error.response?.data
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Prefill mode: ${ENABLE_PREFILL ? 'ENABLED' : 'DISABLED'}`);
});
