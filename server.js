// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Increase timeouts for slow models
app.use((req, res, next) => {
  req.setTimeout(300000);
  res.setTimeout(300000);
  next();
});

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Cache for validated models
const validatedModels = new Set();

// REASONING DISPLAY TOGGLE
const SHOW_REASONING = true;

// Models that require special content format
const ARRAY_CONTENT_MODELS = [
  'moonshotai/kimi-k2.5-instruct',
  'moonshotai/kimi-k2-instruct-0905'
];

// THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = false;

// Models that support thinking parameter
const THINKING_MODELS = [
  'qwen/qwen3-next-80b-a3b-thinking',
  'qwen/qwq-32b-preview'
];

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2.5-instruct',
  'gpt-4-turbo-preview': 'moonshotai/kimi-k2.5-instruct',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o-mini': 'meta/llama-3.1-70b-instruct',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'claude-3-5-sonnet': 'openai/gpt-oss-120b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'kimi-k2.5': 'moonshotai/kimi-k2.5-instruct',
  'kimi': 'moonshotai/kimi-k2.5-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint
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

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    console.log(`Received request for model: ${model}`);
    
    let nimModel = MODEL_MAPPING[model];
    
    if (nimModel) {
      console.log(`Mapped to: ${nimModel}`);
    } else {
      console.log(`Model ${model} not in mapping, trying direct...`);
      
      if (validatedModels.has(model)) {
        nimModel = model;
        console.log(`Using cached validated model: ${model}`);
      } else {
        try {
          const testResponse = await axios.post(`${NIM_API_BASE}/chat/completions`, {
            model: model,
            messages: [{ role: 'user', content: 'test' }],
            max_tokens: 1
          }, {
            headers: { 
              'Authorization': `Bearer ${NIM_API_KEY}`, 
              'Content-Type': 'application/json' 
            },
            timeout: 5000,
            validateStatus: (status) => status < 500
          });
          
          if (testResponse.status >= 200 && testResponse.status < 300) {
            nimModel = model;
            validatedModels.add(model);
            console.log(`Direct model validated: ${model}`);
          } else {
            console.log(`Direct model failed with status: ${testResponse.status}`);
          }
        } catch (e) {
          console.log(`Direct model test failed: ${e.message}`);
        }
      }
      
      if (!nimModel) {
        console.log(`Falling back to size-based selection...`);
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
          console.log(`Selected 405B model`);
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
          console.log(`Selected 70B model`);
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
          console.log(`Selected 8B model`);
        }
      }
    }
    
    const requiresArrayFormat = ARRAY_CONTENT_MODELS.some(m => nimModel.includes(m.split('/')[1]));
    
    const transformedMessages = messages.map(msg => {
      if (requiresArrayFormat && typeof msg.content === 'string') {
        return {
          ...msg,
          content: [{ type: 'text', text: msg.content }]
        };
      }
      if (!requiresArrayFormat && Array.isArray(msg.content)) {
        const textContent = msg.content
          .filter(item => item.type === 'text')
          .map(item => item.text)
          .join('\n');
        return {
          ...msg,
          content: textContent
        };
      }
      return msg;
    });
    
    const supportsThinking = THINKING_MODELS.includes(nimModel);
    
    const nimRequest = {
      model: nimModel,
      messages: transformedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };
    
    if (ENABLE_THINKING_MODE && supportsThinking) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    console.log(`Using model: ${nimModel} for request`);
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 120000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        res.write(':keepalive\n\n');
        
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
              if (data.choices && data.choices[0] && data.choices[0].delta) {
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
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message ? choice.message.content || '' : '';
          
          if (SHOW_REASONING && choice.message && choice.message.reasoning_content) {
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
    console.error('Proxy error:', error.message);
    if (error.response && error.response.data) {
      console.error('NVIDIA API error details:', JSON.stringify(error.response.data));
    }
    console.error('Request model was:', req.body.model);
    
    res.status(error.response ? error.response.status : 500).json({
      error: {
        message: (error.response && error.response.data && error.response.data.detail) || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response ? error.response.status : 500
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
  console.log(`Timeout configured: 5 minutes for slow models`);
});
