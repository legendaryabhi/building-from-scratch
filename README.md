# Building From Scratch

![Building From Scratch](/assets/banner.png)
This repo is a collection of How AI systems are built from core foundations to models, AI agents, AI IDEs, and emerging trends.

> Abstractions hide complexity; building reveals it.

### Table of Contents

- [Emerging Trends](#emerging-trends)
- [AI Agents](#ai-agents)
- [Developer Tools / AI IDEs](#developer-tools--ai-ides)
- [Retrieval & Memory](#retrieval--memory)
- [Core Models](#core-models)
- [Foundations](#foundations)
- [Contributing](#contributing)
- [License](#license)

---

### Emerging Trends

#### **1. OpenClaw (The "Life OS" Agent)**

_Concept:_ OpenClaw (often stylized with the lobster emoji) is a proactive, local-first AI assistant. Unlike typical bots that wait for you to prompt them, Clawdbot runs 24/7 on your hardware and messages you first via WhatsApp, Telegram, or Discord.

- **GitHub:** [openclaw/openclaw](https://github.com/openclaw/openclaw) – The official repository. A "Claude with hands" that manages your email, calendar, and system files.
- Visit Site [here](https://openclaw.ai)
- [Documentation](https://docs.openclaw.ai/start/getting-started)

[NanoBot](https://www.nanobot.ai) - An Open Source framework for building MCP agents—complete with reasoning, system prompts, tool orchestration, and rich MCP-UI support.

### AI Agents

#### **1. CLI Agents (The Interface)**

Before agents can "think," they need a way to interact with the world. The simplest agent is a CLI tool that parses natural language into command arguments.

- **GitHub:** [simonw/llm](https://github.com/simonw/llm) – The definitive CLI tool for interacting with LLMs. Study how it handles piping (`cat file.txt | llm "summarize"`) and plugin systems.
- **Article:** [Building a CLI Agent with Python](https://www.google.com/search?q=https://simonwillison.net/2023/May/2/llm-cli/) – Simon Willison explains how to treat LLMs as Unix command-line tools. This is essential for understanding how agents can fit into existing developer workflows.
- **YouTube:** [Build a CLI Chatbot with Python & Typer](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DIVwZs5493yA) – Shows how to build the "frontend" of your agent using `Typer`, which is the industry standard for Python CLIs.
- **Notebook:** [Command Line Assistant](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb) – OpenAI's cookbook on using "Function Calling" to turn natural language into CLI commands.

#### **2. AI Agents (Reasoning & Loops)**

This is the core logic: The "ReAct" loop (Reason + Act) where the model pauses, executes code, and observes the output.

- **GitHub:** [pguso/ai-agents-from-scratch](https://github.com/pguso/ai-agents-from-scratch) – **The best resource on this list.** It builds an agent system step-by-step (Introduction → Tools → Memory → Planning) with zero frameworks. It explains _why_ agents are built this way.
- **Article:** [ReAct: Synergizing Reasoning and Acting](https://react-lm.github.io/) – The original blog post that introduced the concept of "Reasoning Traces" (Thought → Action → Observation).
- **YouTube:** [DeepLearning.AI: Functions, Tools and Agents](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) – Even though it uses LangChain, the first module explains the _theory_ of the ReAct loop better than almost any other video.
- **Code Example:** [Simple Python ReAct Loop](https://til.simonwillison.net/llms/python-react-pattern) – A single Python file that implements the ReAct loop using RegEx, showing you don't need a library to parse agent thoughts.

#### **3. Cloud Agents (Async & API Serving)**

Running an agent on a server requires managing long-running tasks. You can't just let an API request "hang" while the agent thinks for 2 minutes.

- **GitHub:** [tiangolo/fastapi](https://github.com/tiangolo/fastapi) – You need to learn FastAPI to serve agents. Focus on the "Background Tasks" section of the docs.
- **Article:** [How to Deploy AI Agents to Production](https://www.google.com/search?q=https://eugeneyan.com/writing/llm-patterns/%23level-3-agents) – Eugene Yan’s guide on the architecture of agent systems: Job Queues, State Persistence, and Failure Recovery.
- **YouTube:** [Asynchronous Task Queues (Celery + Redis)](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DTHxCy-6Enns) – Agents are slow. This tutorial teaches the architecture (Queue → Worker) needed to run agents without timing out your server.
- **Concept:** **Webhooks**. Your agent needs to "call back" your frontend when it finishes a task. Research "Webhooks in Python" to understand this pattern.

#### **4. Multi-Agent Systems (Orchestration)**

How to make two agents talk to each other without them getting stuck in an infinite loop of "Hello" "Hi".

- **GitHub:** [scrapegraphai/ScrapeGraphAI](https://www.google.com/search?q=https://github.com/VinciGit00/ScrapeGraphAI) – A library that uses a graph of agents to scrape websites. The code is readable and shows how to pass state between a "Fetcher Agent" and a "Summary Agent."
- **Article:** [How to Create Multi-Agent Systems Without Frameworks](https://scrapegraphai.com/blog/how-to-create-agent-without-frameworks) – A brilliant tutorial that builds a "Researcher" and "Analyst" agent using raw Python/OpenAI calls, explicitly avoiding LangGraph/CrewAI.
- **YouTube:** [Multi-Agent Collaboration with AutoGen](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3D1yv-u9Jv8CQ) – Andrew Ng’s breakdown of _Conversation Patterns_ (Sequential vs. Hierarchical chat).
- **Notebook:** [Camel Role-Playing](https://github.com/camel-ai/camel/blob/master/examples/ai_society/role_playing.py) – One of the first papers on multi-agent roleplay. The code shows how to prompt-engineer "Inception" (getting agents to instruct each other).

#### **5. Autonomous Agents (Self-Directed)**

Agents that create their own tasks. This is the logic behind "BabyAGI" and "AutoGPT."

- **GitHub:** [yoheinakajima/babyagi](https://github.com/yoheinakajima/babyagi) – The original 100-line script. It is the cleanest implementation of the `Task Creation` → `Task Prioritization` → `Execution` loop.
- **Article:** [AutoGPT Logic Explained](https://www.google.com/search?q=https://pbs.twimg.com/media/Ftqlb_aXwAIsF9i%3Fformat%3Djpg%26name%3Dlarge) – (Image/Diagram) Visualizing the infinite loop of an autonomous agent.
- **YouTube:** [Building an Autonomous Agent in Python](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3D5qQHHccrOGs) – Explains the "stack" data structure used to manage the agent's todo list.
- **Project:** Build a script that takes a goal ("Research the history of pizza"), breaks it into 3 sub-tasks, executes them, and writes a report to a file.

#### **6. Agent Evaluation (Safety & Metrics)**

How do you know if your agent is stuck in a loop or hallucinating?

- **GitHub:** [ukgovernmentbeis/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) – An evaluation framework by the UK AI Safety Institute. It’s code-heavy but shows the standard for testing agent safety and capability.
- **Article:** [Evaluation of LLM Agents](https://www.google.com/search?q=https://huyenchip.com/2023/04/11/llm-engineering.html%23eval) – Chip Huyen explains why "Unit Tests" don't work for agents and introduces "Evals" (using an LLM to grade another LLM).
- **YouTube:** [Eval Driven Development](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3D4W1t8e3q7f4) – Teaches you to write the "Test Case" (e.g., "Did the agent actually book the flight?") _before_ you write the agent.
- **Metric:** **Pass@k**. If you run your agent 10 times on the same task, how many times does it succeed? This is the only metric that matters for agents.

### Developer Tools / AI IDEs

#### **1. AI-Powered IDEs / Copilots**

_(How AI becomes a real coding partner, not just autocomplete)_

- **GitHub:** [Continue](https://github.com/continuedev/continue) – Open-source AI IDE extension showing how context gathering, prompt construction, and model calls are wired together inside an editor.
- **GitHub:** [Zed (AI features)](https://github.com/zed-industries/zed) – Modern open-source editor with collaborative + AI hooks; great reference for building AI-native editors.
- **Article:** [How AI Coding Assistants Work Under the Hood](https://blog.continue.dev/how-continue-works) – Explains prompt assembly, context windows, file selection, and tool calling in IDE copilots.
- **YouTube:** [Building an AI Coding Assistant (from scratch concepts)](https://www.youtube.com/watch?v=NeT1zryaeDM) – Walkthrough of how copilots reason over code, not just generate text.
- **Notebook / Example:** [Continue Prompt Templates](https://github.com/continuedev/continue/tree/main/prompts) – Real prompt “programs” used inside an IDE.

**Why this matters:**
Copilots are _agents embedded into developer workflows_ — understanding them teaches context retrieval, prompt compilation, and iterative reasoning.

#### **2. Prompt Tooling / Prompt “Compilers”**

_(Turning natural language into structured agent programs)_

- **GitHub:**[LangChain (Educational examples)](https://github.com/langchain-ai/langchain) – Focus on how prompts are chained, parsed, and executed (ignore the abstraction hype, read the internals).
- **GitHub:**[Prompt Engineering Guide (Repo)](https://github.com/dair-ai/Prompt-Engineering-Guide) – Clear mental models for prompt structure, roles, constraints, and failure modes.
- **Article:** [Prompts Are Programs](https://www.promptingguide.ai/techniques) – Treats prompts as executable logic, not text hacks.
- **YouTube:** [Prompt Engineering for Agents (Deep Dive)](https://www.youtube.com/watch?v=dOxUroR57xs) – Shows how prompts evolve into planning + execution systems.
- **Notebook:** [LangChain Agent Notebooks](https://github.com/langchain-ai/langchain/tree/master/docs/docs/modules/agents) – Minimal notebooks showing ReAct, tool-calling, and memory loops.

**Why this matters:**
Agents don’t “chat” — they **execute prompt-compiled logic**. This is the compiler layer of AI systems.

#### **3. Agent Dashboards & Cowork-Style Playgrounds**

_(Claude Cowork-like systems — but open source)_

- **GitHub:** [AionUi](https://github.com/iOfficeAI/AionUi) – Open-source **Claude-Cowork alternative**: multi-agent sessions, local models, file access, task execution, and UI orchestration.
- **GitHub:** [CrewBench](https://github.com/CrewBench/CrewBench) – Desktop control layer for AI coding agents; shows how agents are managed, logged, and replayed.
- **GitHub:** [Eigent](https://github.com/eigent-ai/eigent) – Open-source local AI agent system with task execution and plugin-style actions.
- **Article:** [What Is an AI Cowork Environment?](https://aionui.com/docs/architecture) – Explains how cowork UIs coordinate agents, tools, memory, and humans.
- **YouTube:** [Building AI Agent Dashboards](https://www.youtube.com/watch?v=H1pYtG5y7rM) – Visual overview of agent control panels and orchestration layers.
- **Example Project:** [AionUi Demo Projects](https://github.com/iOfficeAI/AionUi/tree/main/examples) – Real examples of multi-agent workflows.

**Why this matters:**
Cowork systems show **how agents are supervised, debugged, and composed**, which is essential for real-world AI systems.

---

#### **4. Code + Notebook Integrations**

_(Where agents meet experimentation, data, and iteration)_

- **GitHub:** [Jupyter AI](https://github.com/jupyterlab/jupyter-ai) – Open-source integration of LLMs into notebooks, showing how agents assist exploration.
- **GitHub:** [AutoGen Studio](https://github.com/microsoft/autogen) – Multi-agent conversations + notebooks for debugging agent behavior step-by-step.
- **Article:** [Agents in Notebooks: Why It Works](https://blog.jupyter.org/introducing-jupyter-ai-8d8a3d47b0c7) – Explains how notebooks become agent playgrounds.
- **YouTube:** [Building Multi-Agent Systems in Jupyter](https://www.youtube.com/watch?v=V0Y6Q0zvF6g) – Hands-on agent orchestration with notebooks.
- **Notebook:** [AutoGen Agent Examples](https://github.com/microsoft/autogen/tree/main/notebook) – Runnable notebooks showing agent roles, tools, and conversations.

**Why this matters:**
Notebooks are the **lab bench for agents** — perfect for inspecting reasoning, memory, and failure modes.

### Retrieval & Memory

#### Vector Representations & Embeddings

Before you can search, you must turn text into numbers. Understanding _how_ these vectors capture meaning is the first step.

- **Article:** [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) – Jay Alammar again. This is the visual bible for understanding how words become vectors.
- **GitHub:** [nicolas-ivanov/tf_word2vec](https://www.google.com/search?q=https://github.com/nicolas-ivanov/tf_word2vec) – A clean, no-nonsense implementation of Word2Vec using basic TensorFlow/Python. It helps you see the "skip-gram" logic.
- **Notebook:** [Word Embeddings from Scratch](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) – PyTorch's official tutorial on implementing N-Gram language modeling to generate embeddings.

#### Similarity Search & Vector Databases

How do you find the "nearest" vector in a haystack of millions? You can’t just loop through them all (that's ). You need Approximate Nearest Neighbors (ANN).

- **Article:** [Building a Vector Database from Scratch in Python](https://medium.com/@vidiptvashist/building-a-vector-database-from-scratch-in-python-6bd683ba5171) – A perfect weekend project. You build a `VectorStore` class using just NumPy and cosine similarity.
- **Theory (HNSW):** [Hierarchical Navigable Small Worlds (Pinecone)](https://www.pinecone.io/learn/series/faiss/hnsw/) – HNSW is the algorithm behind almost every modern vector DB (Pinecone, Milvus, Chroma). This visualizes the "highway" graphs used for search.
- **Library (The Standard):** [facebookresearch/faiss](https://github.com/facebookresearch/faiss) – Not "from scratch," but studying the `IndexFlatL2` vs `IndexIVFFlat` documentation is essential for understanding quantization and indexing.

#### Retrieval-Augmented Generation (RAG)

The architecture that connects your database to your LLM.

- **Article:** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) – The original paper. It’s surprisingly readable and defines the standard architecture.
- **GitHub:** [ask-my-pdf (Minimal RAG)](https://github.com/mayooear/gpt4-pdf-chatbot-langchain) – While it uses LangChain, look at the `ingest.py` file. It shows the flow: Load PDF → Split Text → Embed → Store → Query.
- **Tutorial:** [Build your own RAG without LangChain](https://www.youtube.com/watch?v=tcqEUSNCn8I) – Excellent video showing how to wire OpenAI API + a local vector store using just Python lists and dictionaries.

#### Memory in Agents

How to make an AI "remember" a conversation from yesterday.

- **Research Paper:** [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) – The famous "Smallville" paper. Section 3 ("Agent Architecture") is the **gold standard** for designing a memory stream (Recency, Importance, Relevance).
- **GitHub:** [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents) – The official code for the paper above. Look at `memory.py` to see how they score memories to decide what to retrieve.
- **Article:** [Memory in LLM Agents](https://www.google.com/search?q=https://www.promptingguide.ai/research/llm-agents%23memory) – A concise breakdown of Short-term (Context) vs Long-term (Vector DB) memory.

#### Long Context / Attention Scaling

How do we make models read entire books? You need better positional embeddings than just 1, 2, 3...

- **Article (RoPE):** [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) – RoPE is used in Llama, GPT-NeoX, and PaLM. This article explains the complex number rotation math intuitively.
- **GitHub:** [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) – A standalone, copy-pasteable implementation of RoPE.
- **Article (ALiBi):** [ALiBi: Train Short, Test Long](https://www.google.com/search?q=https://ofir.io/train_short_test_long.html) – Explains how to bias attention based on distance, allowing models to handle longer text than they were trained on.

### Core Models

#### **Large Language Models (LLMs)**

From counting words to predicting the future. This is the progression from simple statistics to the Transformer architecture.

#### **Bigram / N-gram Models**

- **GitHub:** [karpathy/makemore](https://github.com/karpathy/makemore) – The absolute gold standard. It starts with a Bigram model and evolves into an MLP, RNN, and LSTM.
- **Article:** [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) – A classic read that predates Transformers but explains _why_ language modeling works.
- **Notebook:** [Building a Bigram Language Model](https://www.google.com/search?q=https://www.kaggle.com/code/prashant111/bag-of-words-bigram-model) – A simple, hands-on Kaggle notebook for understanding word probabilities.

#### **Mini GPT / Transformers**

- **GitHub:** [karpathy/minGPT](https://github.com/karpathy/minGPT) – A more educational, readable version of the GPT architecture than the specialized `nanoGPT`.
- **YouTube:** [Coding a Transformer from scratch on paper and in code](https://www.youtube.com/watch?v=ISNdQcPhsts) – A brilliant, slower-paced tutorial that builds the Attention mechanism line-by-line.
- **Article:** [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) – The famous Harvard NLP guide that implements the "Attention Is All You Need" paper with side-by-side PyTorch code.

#### **Inference Loops (KV Cache)**

- **Article:** [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) – Explains the math of inference (KV caching, memory bandwidth) which is critical for making your model fast.
- **GitHub:** [calvin-mccoy/transformer-inference](https://www.google.com/search?q=https://github.com/calvin-mccoy/transformer-inference) – A minimal implementation focused specifically on the sampling loop and KV-caching.

#### **Code Models / Completion Models**

_Note: Code models are essentially LLMs trained on code datasets (like Python source files) rather than English text._

- **GitHub:** [salesforce/CodeGen](https://github.com/salesforce/CodeGen) – While a large repo, looking at their data preprocessing (tokenizing code) is invaluable.
- **Tutorial:** [How to Fine-Tune a Language Model on Code](https://huggingface.co/docs/transformers/tasks/language_modeling) – Using the Hugging Face ecosystem to specialize a base model (like DistilGPT2) for code completion.
- **Dataset:** [The Stack (BigCode)](https://huggingface.co/datasets/bigcode/the-stack) – Not a model, but the resource you need to build a code model. It’s a massive collection of permissive source code.

#### **Multimodal Models**

Connecting text to pixels and audio waveforms.

#### **Vision Transformers (ViT)**

- **GitHub:** [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) – Phil Wang (lucidrains) is a legend for implementing papers days after release. This is the cleanest ViT implementation in existence.
- **YouTube:** [Vision Transformer from Scratch in PyTorch](https://www.youtube.com/watch?v=ovB0ddFtzzA) – A step-by-step implementation that shows how to chop images into "patches" (tokens).

#### **Audio / Speech Transformers**

- **Article:** [Audio Spectrogram Transformer (AST) Explained](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer) – Explains how audio is converted into visual spectrograms so standard Transformers can process it.
- **Notebook:** [Speech Command Classification with Transformers](https://keras.io/examples/audio/transformer_asr/) – A Keras/TensorFlow guide to building a "hearing" model from scratch.

#### **Text + Image (CLIP)**

- **GitHub:** [moein-shariatnia/OpenAI-CLIP](https://github.com/moein-shariatnia/OpenAI-CLIP) – A dedicated repository that recreates the CLIP training loop (matching images to text captions) from scratch.
- **Article:** [Simple Implementation of OpenAI CLIP](https://www.google.com/search?q=https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-computational-perspective-2f35284212e6) – Walks through the "Contrastive Loss" function, which is the magic glue between text and images.

#### **Generative Models**

The "Creative" side of AI.

#### **GANs (Generative Adversarial Networks)**

- **GitHub:** [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) – Single-file implementations of almost every GAN variant (DCGAN, CycleGAN, Pix2Pix).
- **YouTube:** [Build a GAN in 10 Minutes](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DO1sGR_XqsYs) – A rapid-fire simplified guide to the generator-discriminator game.

#### **VAEs (Variational Autoencoders)**

- **Article:** [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) – Explains the "Reparameterization Trick," which is the hardest mathematical concept in VAEs.
- **GitHub:** [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) – High-quality, modular implementations of VAEs.

#### **Diffusion Models (Stable Diffusion Basics)**

- **Article:** [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) – **The single best resource for this topic.** It implements the "Denoising Diffusion Probabilistic Models" (DDPM) paper from scratch with line-by-line commentary.
- **YouTube:** [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) – Explains the physics-inspired math (thermodynamics) behind diffusion without getting lost in jargon.
- **Notebook:** [Classic DDPM from Scratch](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm) – Part of the "LabML Annotated Papers" series, which is excellent for code-first learners.

### Foundations

#### **Mathematics & Stats**

The "logic" of AI. You need to understand how data is transformed and how "learning" is quantified.

- **GitHub:** [dair-ai/Mathematics-for-ML](https://github.com/dair-ai/Mathematics-for-ML) – A massive, community-curated collection of papers and notes for AI math.
- **Article:** [The Matrix Calculus You Need for Deep Learning](https://explained.ai/matrix-calculus/) – A brilliant, visual guide by Jeremy Howard and Terence Parr that bridges the gap between simple calculus and neural networks.
- **YouTube:** [3Blue1Brown – Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) – The absolute best visual intuition for vectors and matrices.
- **Notebook:** [Mathematics for Machine Learning (DeepLearning.AI)](https://www.google.com/search?q=https://github.com/m-bashari/Mathematics-for-Machine-Learning-and-Data-Science-Specialization) – Lab notebooks that apply linear algebra and calculus using Python/NumPy.

#### **Programming Basics**

While Python is the industry standard, seeing AI concepts in JavaScript or pseudocode can demystify the "magic."

- **GitHub:** [trekhleb/homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning) – Python implementations of popular ML algorithms with clear, step-by-step explanations.
- **Article:** [Python for Beginners (Real Python)](https://realpython.com/python-first-steps/) – Deep dives into data types and loops, specifically the "Pythonic" way.
- **YouTube:** [The Coding Train – Intelligence and Learning](https://www.google.com/search?q=https://www.youtube.com/playlist%3Flist%3DPLRqwX-V7Uu6Y7VfR-vS4Wp0f9uN16t5-u) – Daniel Shiffman builds simple neural networks and genetic algorithms from scratch using **JavaScript (p5.js)**.
- **Notebook:** [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) – Interactive notebooks covering everything from NumPy arrays to basic plotting.

#### **Data Structures & Algorithms**

AI is essentially a series of search and optimization problems on complex data structures like Graphs and Trees.

- **GitHub:** [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) – The largest open-source resource for learning how to code every algorithm imaginable from scratch.
- **Article:** [Introduction to A\* Search](https://www.redblobgames.com/pathfinding/a-star/introduction.html) – The best interactive article on graph search and heuristics.
- **YouTube:** [MIT 6.006 – Introduction to Algorithms](https://www.google.com/search?q=https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/video-lectures/) – Classic academic rigor for understanding trees, heaps, and graphs.
- **Notebook:** [Algorithms in Jupyter](https://github.com/aimacode/aima-python) – Implementations for "Artificial Intelligence: A Modern Approach" (the definitive AI textbook).

#### **Foundations of Machine Learning**

This is the "Transformer era" toolkit. Modern AI lives and dies by how it represents and attends to data.

- **GitHub:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) – The cleanest, most educational implementation of a GPT-style Transformer in about 300 lines of PyTorch.
- **Article:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) – Jay Alammar’s legendary visual breakdown of attention and embeddings.
- **YouTube:** [Andrej Karpathy – Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) – A 2-hour masterclass that builds a Transformer from a blank file.
- **Notebook:** [Transformers from Scratch (Peter Bloem)](https://www.google.com/search?q=https://github.com/pbloem/transformers) – A minimal, clear PyTorch implementation of the "Attention is All You Need" architecture.

#### **Optional Extras (Optimization)**

How models actually improve. Gradient Descent is the "engine" of AI training.

- **GitHub:** [karpathy/micrograd](https://github.com/karpathy/micrograd) – A tiny scalar-valued autograd engine. It’s the best way to see how backpropagation actually works.
- **Article:** [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/) – Sebastian Ruder’s definitive guide on Adam, RMSProp, and momentum.
- **YouTube:** [StatQuest: Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) – Josh Starmer makes complex optimization feel intuitive and simple.
- **Notebook:** [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) – A hands-on look at how automatic differentiation (the core of loss optimization) works.

### Contributing

We welcome contributions from the community. If you have a resource to add or a correction to make, please review our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to propose changes.

---

### License

This repository is licensed under the [MIT License](LICENSE).
