const mainTextInput = document.getElementById('main-text');
const submitTextButton = document.getElementById('submit-text');
const sentenceBoundariesCheckbox = document.getElementById('sentence-boundaries-checkbox');
const queryTextInput = document.getElementById('query-text');
const submitQueryButton = document.getElementById('submit-query');
const statusMessage = document.getElementById('status-message');
const indexingTimeMessage = document.getElementById('indexing-time-message');
const progressBarContainer = document.getElementById('progress-bar-container');
const progressBarFill = document.getElementById('progress-bar-fill');
const progressMessage = document.getElementById('progress-message');
const loader = document.getElementById('loader');
const queryLoader = document.getElementById('query-loader');
const resultsOutput = document.getElementById('results-output');
const querySection = document.getElementById('query-section');
const resultsSection = document.getElementById('results-section');
const searchTimeMessage = document.getElementById('search-time-message');
const themeToggleButton = document.getElementById('theme-toggle');
const themeIconLight = document.getElementById('theme-icon-light');
const themeIconDark = document.getElementById('theme-icon-dark');
const qaytSwitch = document.getElementById('qayt-switch');
const qaytStatusInfo = document.getElementById('qayt-status-info');

let embedderInstance;
let indexedSegments = []; // Will store { text: string, embedding: number[] }
let initialMainTextValue = ""; 
let lastIndexedWithSentenceBoundaries = sentenceBoundariesCheckbox.checked;
let queryAsYouTypeEnabled = localStorage.getItem('qaytEnabled') === 'true';
let indexedMainTextWordCount = 0; 

const MAX_WORDS_IN_MAIN_TEXT_FOR_QAYT = 5000;
const MAX_WORDS_FOR_SEGMENT_SOURCE = 1_000_000_000;
const MAX_SEGMENTS_TO_GENERATE = 1_000_000_000;
const EMBEDDING_CHUNK_SIZE = 1000;
const MAX_RESULTS_DISPLAY = 20;
const QAYT_DEBOUNCE_DELAY = 0; 

function applyTheme(isDark) {
    if (isDark) {
        document.documentElement.classList.add('dark');
        document.documentElement.classList.remove('light');
        themeIconLight.classList.remove('hidden');
        themeIconDark.classList.add('hidden');
    } else {
        document.documentElement.classList.remove('dark');
        document.documentElement.classList.add('light');
        themeIconLight.classList.add('hidden');
        themeIconDark.classList.remove('hidden');
    }
}

const storedTheme = localStorage.getItem('theme');
const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
let currentThemeIsDark = storedTheme ? storedTheme === 'dark' : systemPrefersDark;
applyTheme(currentThemeIsDark);

themeToggleButton.addEventListener('click', () => {
    currentThemeIsDark = !currentThemeIsDark;
    applyTheme(currentThemeIsDark);
    localStorage.setItem('theme', currentThemeIsDark ? 'dark' : 'light');
});

qaytSwitch.checked = queryAsYouTypeEnabled;
function updateQaytUI() {
    if (queryAsYouTypeEnabled && indexedMainTextWordCount >= MAX_WORDS_IN_MAIN_TEXT_FOR_QAYT) {
        qaytStatusInfo.textContent = `(Preference: ON, but inactive: source text > ${MAX_WORDS_IN_MAIN_TEXT_FOR_QAYT-1} words)`;
    } else if (queryAsYouTypeEnabled) {
        qaytStatusInfo.textContent = '(Active)';
    } else {
        qaytStatusInfo.textContent = `(Preference: OFF)`;
    }
    const qaytEffective = queryAsYouTypeEnabled && indexedMainTextWordCount < MAX_WORDS_IN_MAIN_TEXT_FOR_QAYT && indexedSegments.length > 0;
    submitQueryButton.style.display = qaytEffective ? 'none' : 'inline-block';

}
updateQaytUI(); 

qaytSwitch.addEventListener('change', () => {
    queryAsYouTypeEnabled = qaytSwitch.checked;
    localStorage.setItem('qaytEnabled', queryAsYouTypeEnabled);
    updateQaytUI();
});

function debounce(func, delay) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
}

async function createEmbedder(model_name = "minishlab/potion-retrieval-32M", options = {}) {
    if (!window.transformers) {
         throw new Error("Transformers.js is not loaded.");
    }
    const { AutoModel, AutoTokenizer, Tensor } = window.transformers;
    const { progress_callback = null } = options;

    statusMessage.textContent = 'Loading embedding model...';
    loader.classList.remove('hidden');
    submitTextButton.disabled = true;

    try {
        const tokenizer = await AutoTokenizer.from_pretrained(model_name, { progress_callback });
        const model = await AutoModel.from_pretrained(model_name, {
            config: { model_type: 'model2vec' }, dtype: 'fp32', progress_callback,
        });
        
        statusMessage.textContent = 'Embedding model loaded!';
        loader.classList.add('hidden');
        if (mainTextInput.value.trim()) submitTextButton.disabled = false;

        return async function embedTexts(textsToEmbed) {
            if (!textsToEmbed || textsToEmbed.length === 0) return [];
            const tokenizedResult = await tokenizer(textsToEmbed, {
                add_special_tokens: false, return_tensor: false, padding: false, 
                truncation: true, max_length: tokenizer.model_max_length || 512 
            });
            const token_ids_per_text = tokenizedResult.input_ids;
            const offsets = [0];
            for (let i = 0; i < token_ids_per_text.length - 1; i++) {
                offsets.push(offsets[i] + token_ids_per_text[i].length);
            }
            const flattened_input_ids = token_ids_per_text.flat();
            if (flattened_input_ids.length === 0 && textsToEmbed.length > 0) return textsToEmbed.map(() => []);
            if (flattened_input_ids.length === 0 && textsToEmbed.length === 0) return [];
            const model_inputs = {
                input_ids: new Tensor("int64", BigInt64Array.from(flattened_input_ids.map(BigInt)), [flattened_input_ids.length]),
                offsets: new Tensor("int64", BigInt64Array.from(offsets.map(BigInt)), [offsets.length]),
            };
            const { embeddings } = await model(model_inputs);
            return embeddings.tolist();
        };
    } catch (error) {
        console.error("Error loading model/tokenizer:", error);
        statusMessage.textContent = `Error loading model: ${error.message}. Try refreshing.`;
        statusMessage.classList.add("text-red-500", "dark:text-red-400");
        loader.classList.add('hidden');
        submitTextButton.disabled = true; 
        throw error;
    }
}

async function initializeEmbedder() {
    if (!embedderInstance) {
        try {
            embedderInstance = await createEmbedder("minishlab/potion-base-8M", {
                progress_callback: (data) => {
                    if (data.status === 'progress') {
                        const percent = (data.loaded / data.total * 100).toFixed(1);
                        statusMessage.textContent = `Loading model: ${data.file} (${percent}%)`;
                    } else { statusMessage.textContent = `Model status: ${data.status} - ${data.file || ''}`; }
                }
            });
            statusMessage.textContent = "Embedder ready.";
            if (mainTextInput.value.trim()) submitTextButton.disabled = false;
        } catch (e) {
            statusMessage.textContent = `Failed to initialize embedder: ${e.message}.`;
            statusMessage.classList.add("text-red-500", "dark:text-red-400");
            submitTextButton.disabled = true; submitQueryButton.disabled = true;
        }
    }
}

document.addEventListener('transformersReady', initializeEmbedder);
if (window.transformers && !embedderInstance) initializeEmbedder();

function resetApplicationStateDueToChange(customMessage = "Source text or segmentation method changed. Re-process.") {
    indexedSegments = [];
    indexedMainTextWordCount = 0;
    querySection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    resultsOutput.innerHTML = '<p class="text-custom-gray-500 dark:text-custom-gray-400">Perform a new search.</p>';
    statusMessage.textContent = customMessage;
    statusMessage.classList.remove("text-red-500", "dark:text-red-400", "text-yellow-500", "dark:text-yellow-400");
    indexingTimeMessage.textContent = "";
    progressMessage.textContent = "";
    progressBarContainer.classList.add('hidden');
    progressBarFill.style.width = '0%';
    searchTimeMessage.textContent = "";
    queryTextInput.value = '';
    queryTextInput.disabled = true;
    submitQueryButton.disabled = true;
    if (embedderInstance && mainTextInput.value.trim()) submitTextButton.disabled = false;
    updateQaytUI();
}

mainTextInput.addEventListener('input', () => {
    if (mainTextInput.value !== initialMainTextValue && indexedSegments.length > 0) {
        resetApplicationStateDueToChange("Source text changed. Re-process.");
    }
    submitTextButton.disabled = !(embedderInstance && mainTextInput.value.trim());
});

sentenceBoundariesCheckbox.addEventListener('change', () => {
    if (indexedSegments.length > 0 && sentenceBoundariesCheckbox.checked !== lastIndexedWithSentenceBoundaries) {
        resetApplicationStateDueToChange("Segmentation method changed. Re-process.");
    }
});

function getSentencesFromText(text) {
    if (!text) return [];
    const marker = "<SENTENCE_END_MARKER_XYZ>"; // Unique marker
    // Replace sentence terminators followed by optional space with terminator + marker + original space
    let processedText = text.replace(/([.?!])(\s*)/g, `$1${marker}$2`);
    let sentences = processedText.split(marker);
    return sentences.map(s => s.trim()).filter(s => s.length > 0);
}


submitTextButton.addEventListener('click', async () => {
    const mainText = mainTextInput.value.trim();
    initialMainTextValue = mainTextInput.value; 
    lastIndexedWithSentenceBoundaries = sentenceBoundariesCheckbox.checked;

    if (!mainText) { statusMessage.textContent = "Please enter text."; return; }
    if (!embedderInstance) { 
        statusMessage.textContent = "Embedder not ready. Wait or refresh."; 
        if (window.transformers) await initializeEmbedder(); 
        if (!embedderInstance) return; 
    }

    submitTextButton.disabled = true;
    queryTextInput.disabled = true;
    submitQueryButton.disabled = true;
    loader.classList.add('hidden'); 
    progressBarContainer.classList.remove('hidden');
    progressBarFill.style.width = '0%';
    progressMessage.textContent = 'Preparing segments...';
    resultsSection.classList.add('hidden');
    resultsOutput.innerHTML = '<p class="text-custom-gray-500 dark:text-custom-gray-400">Processing...</p>';
    statusMessage.textContent = "Generating text segments...";
    indexingTimeMessage.textContent = "";
    statusMessage.classList.remove("text-red-500", "dark:text-red-400", "text-yellow-500", "dark:text-yellow-400");

    const startTime = performance.now();
    const overallWords = mainText.split(/\s+/).filter(w => w.length > 0);
    indexedMainTextWordCount = overallWords.length;

    let combinations = [];
    const useSentenceBoundaries = sentenceBoundariesCheckbox.checked;

    if (useSentenceBoundaries) {
        const sentences = getSentencesFromText(mainText);
        sentences.forEach(sentence => {
            const words = sentence.split(/\s+/).filter(w => w.length > 0);
            for (let i = 0; i < words.length; i++) {
                for (let j = i; j < words.length; j++) {
                    combinations.push(words.slice(i, j + 1).join(" "));
                }
            }
        });
    } else {
         const words = mainText.split(/\s+/).filter(w => w.length > 0);
         for (let i = 0; i < words.length; i++) {
            for (let j = i; j < words.length; j++) {
               combinations.push(words.slice(i, j + 1).join(" "));
            }
        }
    }

    let uniqueCombinations = Array.from(new Set(combinations));
    if (uniqueCombinations.length > MAX_SEGMENTS_TO_GENERATE) {
        uniqueCombinations = uniqueCombinations.slice(0, MAX_SEGMENTS_TO_GENERATE);
        statusMessage.textContent = `Reached max unique segments limit (${MAX_SEGMENTS_TO_GENERATE}). Processing these.`;
        statusMessage.classList.add("text-yellow-500", "dark:text-yellow-400");
    }
    
    if (indexedMainTextWordCount > MAX_WORDS_FOR_SEGMENT_SOURCE && !useSentenceBoundaries && !statusMessage.textContent.startsWith("Reached max")) {
         const currentMsg = statusMessage.textContent ? statusMessage.textContent + " Also, " : "";
         statusMessage.textContent = currentMsg + `Warning: Input text is very long (${indexedMainTextWordCount} words) and sentence boundaries are off. This may be slow or generate many segments.`;
         if(!statusMessage.classList.contains("text-yellow-500")) {
            statusMessage.classList.add("text-yellow-500", "dark:text-yellow-400");
         }
    }
    
    if (uniqueCombinations.length === 0) {
        statusMessage.textContent = "No unique text segments generated.";
        progressBarContainer.classList.add('hidden');
        progressMessage.textContent = "";
        submitTextButton.disabled = false;
        return;
    }
    
    statusMessage.textContent = `Embedding ${uniqueCombinations.length} unique segments...`;
    
    const allEmbeddings = [];
    let segmentsProcessed = 0;
    indexedSegments = []; // Clear previous

    try {
        for (let i = 0; i < uniqueCombinations.length; i += EMBEDDING_CHUNK_SIZE) {
            const chunkTexts = uniqueCombinations.slice(i, i + EMBEDDING_CHUNK_SIZE);
            if (chunkTexts.length === 0) continue;

            const chunkEmbeddings = await embedderInstance(chunkTexts);
            
            for(let k=0; k < chunkTexts.length; k++){
                if(chunkEmbeddings[k] && chunkEmbeddings[k].length > 0) {
                    indexedSegments.push({ text: chunkTexts[k], embedding: chunkEmbeddings[k] });
                } else {
                    console.warn(`Segment "${chunkTexts[k]}" resulted in empty or invalid embedding. Skipping.`);
                }
            }
            segmentsProcessed += chunkTexts.length;

            const progressPercent = Math.min((segmentsProcessed / uniqueCombinations.length) * 100, 100);
            progressBarFill.style.width = `${progressPercent}%`;
            progressMessage.textContent = `Embedding unique segment ${Math.min(segmentsProcessed, uniqueCombinations.length)} of ${uniqueCombinations.length}...`;
            await new Promise(resolve => setTimeout(resolve, 0)); // Yield for UI update
        }


        if(indexedSegments.length === 0 && uniqueCombinations.length > 0) {
             throw new Error("All unique segments resulted in invalid or empty embeddings.");
        }

        const endTime = performance.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        indexingTimeMessage.textContent = `Indexing ${indexedSegments.length} segments took ${duration}s.`;
        statusMessage.textContent = `Indexed ${indexedSegments.length} unique segments. Ready for queries.`;
        
        querySection.classList.remove('hidden');
        querySection.classList.add('animate-slide-up');
        resultsSection.classList.add('hidden');
        resultsOutput.innerHTML = '<p class="text-custom-gray-500 dark:text-custom-gray-400">Enter query...</p>';
        queryTextInput.value = ''; 
        queryTextInput.disabled = false;
        submitQueryButton.disabled = false;
        queryTextInput.focus();
        updateQaytUI();

    } catch (error) {
        statusMessage.textContent = `Error during embedding: ${error.message}`;
        statusMessage.classList.add("text-red-500", "dark:text-red-400");
        console.error("Embedding error:", error);
        submitTextButton.disabled = false; 
    } finally {
        progressBarContainer.classList.add('hidden');
        progressMessage.textContent = "";
        loader.classList.add('hidden');
    }
});

function dotProduct(vecA, vecB) { return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0); }
function magnitude(vec) { return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0)); }
function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length === 0 || vecA.length !== vecB.length) return 0; 
    const prod = dotProduct(vecA, vecB);
    const magA = magnitude(vecA);
    const magB = magnitude(vecB);
    return (magA === 0 || magB === 0) ? 0 : prod / (magA * magB);
}

async function handleQuery() {
    const queryText = queryTextInput.value.trim();
    if (!queryText) {
        if (!resultsSection.classList.contains('hidden') || document.activeElement === queryTextInput) {
            resultsOutput.innerHTML = '<p class="text-custom-gray-500 dark:text-custom-gray-400">Please enter a query.</p>';
            resultsSection.classList.remove('hidden');
            resultsSection.classList.add('animate-slide-up');
            searchTimeMessage.textContent = "";
        }
        return;
    }
    if (!embedderInstance) { statusMessage.textContent = "Embedder not ready."; return; }
    if (indexedSegments.length === 0) { statusMessage.textContent = "No segments indexed."; return; }

    submitQueryButton.disabled = true;
    queryLoader.classList.remove('hidden');
    resultsOutput.innerHTML = '';
    searchTimeMessage.textContent = "";
    const queryStartTime = performance.now();

    try {
        const queryEmbeddingArray = await embedderInstance([queryText]);
        if (!queryEmbeddingArray || queryEmbeddingArray.length === 0 || !queryEmbeddingArray[0] || queryEmbeddingArray[0].length === 0) {
            throw new Error("Failed to generate query embedding.");
        }
        const queryEmbedding = queryEmbeddingArray[0];

        const SIMILARITY_THRESHOLD = 0.2;

        const scoredSegments = indexedSegments.map(segment => ({
            text: segment.text,
            similarity: cosineSimilarity(queryEmbedding, segment.embedding)
        })).filter(segment => segment.similarity >= SIMILARITY_THRESHOLD);

        scoredSegments.sort((a, b) => b.similarity - a.similarity); 

        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('animate-slide-up');

        if (scoredSegments.length === 0) {
            resultsOutput.innerHTML = `<p class="text-custom-gray-600 dark:text-custom-gray-400">No segments found matching your query (threshold ≥ ${SIMILARITY_THRESHOLD*100}%).</p>`;
        } else {
            scoredSegments.slice(0, MAX_RESULTS_DISPLAY).forEach(segment => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'p-4 bg-custom-gray-200 dark:bg-custom-gray-700 rounded-lg shadow hover:shadow-md dark:hover:bg-custom-gray-600 animate-fade-in';
                const textEl = document.createElement('p');
                textEl.className = 'text-sm text-custom-gray-800 dark:text-custom-gray-200';
                textEl.textContent = segment.text;
                const scoreEl = document.createElement('p');
                scoreEl.className = 'text-xs font-semibold mt-1';
                const similarityPercentage = (segment.similarity * 100).toFixed(2);
                let scoreColorClass = 'text-custom-green-500 dark:text-custom-green-500'; 
                if (segment.similarity > 0.85) scoreColorClass = 'text-custom-green-300 dark:text-custom-green-300 font-bold';
                else if (segment.similarity > 0.70) scoreColorClass = 'text-custom-green-400 dark:text-custom-green-400';
                else if (segment.similarity > 0.50) scoreColorClass = 'text-custom-teal-400 dark:text-custom-teal-400';
                scoreEl.className += ` ${scoreColorClass}`;
                scoreEl.textContent = `Similarity: ${similarityPercentage}%`;
                resultDiv.appendChild(textEl); resultDiv.appendChild(scoreEl);
                resultsOutput.appendChild(resultDiv);
            });
             if (scoredSegments.length > MAX_RESULTS_DISPLAY) {
                const noticeEl = document.createElement('p');
                noticeEl.className = 'text-xs text-center text-custom-gray-500 dark:text-custom-gray-400 mt-2';
                noticeEl.textContent = `Displaying top ${MAX_RESULTS_DISPLAY} of ${scoredSegments.length} results.`;
                resultsOutput.appendChild(noticeEl);
            }
        }
        const queryEndTime = performance.now();
        const queryDuration = ((queryEndTime - queryStartTime) / 1000).toFixed(2);
        searchTimeMessage.textContent = `Search took ${queryDuration}s.`;

    } catch (error) {
        resultsOutput.innerHTML = `<p class="text-red-500 dark:text-red-400">Query error: ${error.message}</p>`;
        console.error("Query error:", error);
    } finally {
        submitQueryButton.disabled = false;
        queryLoader.classList.add('hidden');
    }
}
const debouncedHandleQuery = debounce(handleQuery, QAYT_DEBOUNCE_DELAY);

submitQueryButton.addEventListener('click', handleQuery);
queryTextInput.addEventListener('keypress', (event) => { if (event.key === 'Enter') handleQuery(); });

queryTextInput.addEventListener('input', () => {
    const qaytEffective = queryAsYouTypeEnabled && indexedMainTextWordCount < MAX_WORDS_IN_MAIN_TEXT_FOR_QAYT && indexedSegments.length > 0;
    if (qaytEffective) {
        if (queryTextInput.value.trim() === "") { 
            resultsOutput.innerHTML = '<p class="text-custom-gray-500 dark:text-custom-gray-400">Enter query...</p>';
            searchTimeMessage.textContent = "";
        } else {
            debouncedHandleQuery();
        }
    }
});

mainTextInput.value = "Earth Observation is crucial to society, providing vital data for climate change monitoring, effective disaster management, and the optimization of natural resources. Advanced sensor technology and AI-driven analytics are transforming how we interpret this information, leading to more accurate predictions and sustainable practices across various sectors including agriculture, urban planning, and environmental protection.";
initialMainTextValue = mainTextInput.value; 
if(embedderInstance) submitTextButton.disabled = !mainTextInput.value.trim();
else submitTextButton.disabled = true;
updateQaytUI();
