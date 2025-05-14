// Armazena elementos DOM globalmente
let loadingSpinner;
let btnStart;
let btnAbout;
let predictionResult;
// Variável para controle do estado de carregamento
let isLoading = false;
// Objeto de seções para navegação
let sections = {};

document.addEventListener('DOMContentLoaded', function() {
    // Inicializar elementos
    loadingSpinner = document.getElementById('loading-spinner');
    btnStart = document.getElementById('btn-start');
    btnAbout = document.getElementById('btn-about');
    predictionResult = document.getElementById('prediction-result');
    
    // Inicializar seções
    sections = {
        modelos: document.getElementById('modelos'),
        predicao: document.getElementById('predicao'),
        comparacao: document.getElementById('comparacao'),
        about: document.getElementById('about')
    };

    // Funcionalidade do menu hamburger para dispositivos móveis
    const hamburgerBtn = document.querySelector('.hamburger-menu');
    const nav = document.querySelector('nav');

    if (hamburgerBtn) {
        hamburgerBtn.addEventListener('click', () => {
            nav.classList.toggle('active');
            hamburgerBtn.classList.toggle('active');
        });

        // Fechar menu ao clicar fora dele
        document.addEventListener('click', (e) => {
            if (!nav.contains(e.target) && !hamburgerBtn.contains(e.target)) {
                nav.classList.remove('active');
                hamburgerBtn.classList.remove('active');
            }
        });

        // Fechar menu ao clicar em um link
        nav.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                nav.classList.remove('active');
            });
        });
    }
    // Botões de treinamento
    const btnTrainKnn = document.getElementById('btn-train-knn');
    const btnTrainDt = document.getElementById('btn-train-dt');
    const btnTrainRf = document.getElementById('btn-train-rf');
    const btnTrainSvm = document.getElementById('btn-train-svm');
    const btnTrainAll = document.getElementById('btn-train-all');

    // Botões de teste
    const btnTestKnn = document.getElementById('btn-test-knn');
    const btnTestDt = document.getElementById('btn-test-dt');
    const btnTestRf = document.getElementById('btn-test-rf');
    const btnTestSvm = document.getElementById('btn-test-svm');

    // Predição 
    const btnPredict = document.getElementById('btn-predict');
    

    // Comparação
    const btnCompareAll = document.getElementById('btn-compare-all');
    const comparisonResults = document.getElementById('comparison-results');

    // Navegação
    const navLinks = document.querySelectorAll('nav ul li a');
    const allSections = document.querySelectorAll('section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);

            // Atualiza classe active
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');

            // Scroll suave para a seção
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Observador de interseção para atualizar menu
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
                });
            }
        });
    }, { threshold: 0.5 });

    allSections.forEach(section => observer.observe(section));

    // Botões da hero section
    if (btnStart) {
        btnStart.addEventListener('click', function() {
            // Apenas faz o scroll para a seção de modelos sem ocultar outras seções
            sections.modelos.scrollIntoView({ behavior: 'smooth' });
            
            // Atualiza navegação
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#modelos') {
                    link.classList.add('active');
                }
            });
        });
    }

    if (btnAbout) {
        btnAbout.addEventListener('click', function() {
            // Verifica se a seção about está com a classe hidden
            if (sections.about.classList.contains('hidden')) {
                sections.about.classList.remove('hidden');
            }
            
            // Scroll para seção sobre
            sections.about.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // Função para treinar modelo específico
    function trainModel(modelName) {
        showLoading("Treinando modelo, aguarde...");

        const formData = {
            testSize: 0.3,
            randomState: 42
        };

        if (modelName) {
            formData.model = modelName;
        }

        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Falha ao treinar modelo');
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            const modelNames = {
                'knn': 'KNN',
                'decision_tree': 'Árvore de Decisão', 
                'random_forest': 'Random Forest',
                'svm': 'SVM',
                'all': 'Todos os modelos'
            };
            const displayName = modelNames[modelName] || modelName;
            alert(`${displayName} treinado com sucesso!`);
        })
        .catch(error => {
            hideLoading();
            alert('Erro ao treinar modelo: ' + error.message);
        });
    }

    // Função para avaliar modelo
    function testModel(modelName) {
        showLoading("Avaliando modelo, aguarde...");

        const requestData = { model: modelName };

        fetch('/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Falha ao testar modelo');
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            displayModelResults(data, modelName);
        })
        .catch(error => {
            hideLoading();
            alert('Erro ao testar modelo: ' + error.message);
        });
    }

    // Função para exibir resultados do modelo
    function displayModelResults(results, modelName) {
        const resultsContainer = document.getElementById('model-results');
        if (!resultsContainer) return;

        resultsContainer.classList.remove('hidden');

        // Formata o nome do modelo para exibição
        const displayName = modelName === 'knn' ? 'KNN' : 
                            modelName === 'decision_tree' ? 'Árvore de Decisão' :
                            modelName === 'random_forest' ? 'Random Forest' : 
                            modelName === 'svm' ? 'SVM' : modelName;

        // Obtém os dados específicos do modelo
        const modelData = results[modelName];

        if (!modelData) {
            resultsContainer.innerHTML = '<p>Nenhum resultado disponível para este modelo.</p>';
            return;
        }

        let html = `
            <div class="model-results-card">
                <div class="model-header">
                    <h3>Resultados do Modelo: ${displayName}</h3>
                </div>
                
                <div class="row model-content">
                    <div class="col-half">
                        <div class="metrics-panel">
                            <h4><i class="fas fa-chart-line"></i> Métricas de Desempenho</h4>
                            <table class="metrics-table">
                                <thead>
                                    <tr>
                                        <th>Métrica</th>
                                        <th><i class="fas fa-graduation-cap"></i> Treino</th>
                                        <th><i class="fas fa-vial"></i> Teste</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><strong>Acurácia</strong></td>
                                        <td class="metric-value">${(modelData.train.accuracy * 100).toFixed(1)}%</td>
                                        <td class="metric-value test-metric">${(modelData.test.accuracy * 100).toFixed(1)}%</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Precisão</strong></td>
                                        <td class="metric-value">${(modelData.train.precision * 100).toFixed(1)}%</td>
                                        <td class="metric-value test-metric">${(modelData.test.precision * 100).toFixed(1)}%</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Recall</strong></td>
                                        <td class="metric-value">${(modelData.train.recall * 100).toFixed(1)}%</td>
                                        <td class="metric-value test-metric">${(modelData.test.recall * 100).toFixed(1)}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="col-half">
                        <div class="confusion-matrix-panel">
                            <h4><i class="fas fa-th"></i> Matriz de Confusão</h4>
                            <div class="matrix-container">
                                <img src="data:image/png;base64,${modelData.confusion_matrix}" 
                                     alt="Matriz de Confusão" class="img-fluid matrix-image">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Se tiver a superfície de decisão, adiciona
        if (results.decision_boundary) {
            html += `
                <div style="margin-top: 20px;">
                    <h4>Superfície de Decisão</h4>
                    <img src="data:image/png;base64,${results.decision_boundary}" 
                         alt="Superfície de Decisão" class="img-fluid">
                </div>
            `;
        }

        resultsContainer.innerHTML = html;
    }

    // Função para fazer predição
    function predict() {
        // Validar os campos de entrada
        const sepalLength = document.getElementById('sepal_length').value.trim();
        const sepalWidth = document.getElementById('sepal_width').value.trim();
        const petalLength = document.getElementById('petal_length').value.trim();
        const petalWidth = document.getElementById('petal_width').value.trim();
        
        // Validações dos campos
        if (!sepalLength || !sepalWidth || !petalLength || !petalWidth) {
            alert('Por favor, preencha todos os campos de medidas da flor.');
            return;
        }
        
        // Verificar se os valores são numéricos válidos
        const numericFields = [
            { valor: sepalLength, nome: 'Comprimento da Sépala' },
            { valor: sepalWidth, nome: 'Largura da Sépala' },
            { valor: petalLength, nome: 'Comprimento da Pétala' },
            { valor: petalWidth, nome: 'Largura da Pétala' }
        ];
        
        for (const field of numericFields) {
            if (isNaN(parseFloat(field.valor))) {
                alert(`O campo "${field.nome}" deve conter um valor numérico válido.`);
                return;
            }
            
            // Verificar se está no intervalo válido (valores positivos e razoáveis)
            const valor = parseFloat(field.valor);
            if (valor <= 0 || valor > 30) {
                alert(`O valor do campo "${field.nome}" deve estar entre 0 e 30 cm.`);
                return;
            }
        }
        
        // Se passou por todas as validações, mostrar o loading
        showLoading("Realizando predição, aguarde...");

        // Obtém modelo selecionado
        const modelEls = document.querySelectorAll('input[name="model"]');
        let selectedModel = 'knn';

        for (const el of modelEls) {
            if (el.checked) {
                selectedModel = el.value;
                break;
            }
        }

        const requestData = {
            model: selectedModel,
            sepal_length: sepalLength,
            sepal_width: sepalWidth,
            petal_length: petalLength,
            petal_width: petalWidth
        };

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Falha ao fazer predição');
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            displayPrediction(data);
        })
        .catch(error => {
            hideLoading();
            alert('Erro ao fazer predição: ' + error.message);
        });
    }

    // Função para exibir resultado da predição
    function displayPrediction(data) {
        if (!predictionResult) return;

        predictionResult.classList.remove('hidden');

        // Determina classe CSS e imagem baseada na espécie
        let colorClass = 'species-default';
        let speciesImage = '';
        let speciesName = '';
        
        if (data.prediction.toLowerCase().includes('setosa')) {
            colorClass = 'species-setosa';
            speciesImage = 'setosa.svg';
            speciesName = 'Setosa';
        } else if (data.prediction.toLowerCase().includes('versicolor')) {
            colorClass = 'species-versicolor';
            speciesImage = 'versicolor.svg';
            speciesName = 'Versicolor';
        } else if (data.prediction.toLowerCase().includes('virginica')) {
            colorClass = 'species-virginica';
            speciesImage = 'virginica.svg';
            speciesName = 'Virginica';
        }

        // Formata o nome do modelo para exibição
        const modelName = data.model === 'knn' ? 'KNN' : 
                        data.model === 'decision_tree' ? 'Árvore de Decisão' :
                        data.model === 'random_forest' ? 'Random Forest' : 
                        data.model === 'svm' ? 'SVM' : data.model;

        // HTML melhorado com design mais rico e informativo e imagem da espécie
        let html = `
            <div class="prediction-result-card">
                <div class="prediction-header ${colorClass}">
                    <i class="fas fa-leaf prediction-icon"></i>
                    <h3>Resultado da Classificação</h3>
                </div>
                <div class="prediction-body">
                    <div class="prediction-species">
                        <div class="species-image-container">
                            <img src="/static/assets/${speciesImage}" alt="${speciesName}" class="species-image">
                        </div>
                        <h2 class="species-name ${colorClass}">${data.prediction}</h2>
                    </div>
                    <div class="prediction-details">
                        <div class="prediction-item">
                            <i class="fas fa-cogs"></i>
                            <span><strong>Modelo utilizado:</strong> ${modelName}</span>
                        </div>
                        <div class="prediction-item">
                            <i class="fas fa-chart-line"></i>
                            <span><strong>Confiança da previsão:</strong> ${(data.accuracy * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Adiciona gráfico de probabilidades, se disponível
        if (data.probabilities) {
            html += `
                <div class="probabilities-card">
                    <div class="probabilities-header">
                        <h4>Probabilidades por Classe</h4>
                    </div>
                    <div class="probabilities-body">
                        <canvas id="probabilityChart" width="400" height="200"></canvas>
                    </div>
                </div>
            `;

            predictionResult.innerHTML = html;

            // Cria gráfico de probabilidades com cores melhoradas
            createProbabilityChart(data.probabilities);
        } else {
            predictionResult.innerHTML = html;
        }
    }

    // Função para criar gráfico de probabilidades
    function createProbabilityChart(probabilities) {
        const ctx = document.getElementById('probabilityChart');
        if (!ctx) return;

        const labels = Object.keys(probabilities);
        const values = Object.values(probabilities);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probabilidade',
                    data: values,
                    backgroundColor: [
                        '#3D9970',  // setosa
                        '#4A90E2',  // versicolor
                        '#6A1B9A'   // virginica
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // Função para comparar todos os modelos
    function compareModels() {
        showLoading("Comparando modelos, aguarde...");

        fetch('/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: 'all' })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Falha ao comparar modelos');
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            displayComparisonResults(data);
        })
        .catch(error => {
            hideLoading();
            alert('Erro ao comparar modelos: ' + error.message);
        });
    }

    // Função para exibir resultados da comparação
    function displayComparisonResults(results) {
        if (!comparisonResults) return;

        comparisonResults.classList.remove('hidden');

        // Monta a tabela de comparação
        let html = `
            <h3>Comparação de Desempenho dos Modelos</h3>
            <div class="table-responsive">
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Modelo</th>
                            <th>Acurácia (Treino)</th>
                            <th>Acurácia (Teste)</th>
                            <th>Precisão</th>
                            <th>Recall</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            // Adiciona linha para cada modelo
            const modelMap = {
                'knn': 'KNN',
                'decision_tree': 'Árvore de Decisão',
                'random_forest': 'Random Forest',
                'svm': 'SVM'
            };

            Object.keys(results).forEach(modelName => {
                if (modelName !== 'comparison_chart' && modelName !== 'decision_boundary') {
                    const modelData = results[modelName];
                    const displayName = modelMap[modelName] || modelName;

                    html += `
                        <tr>
                            <td class="model-name">${displayName}</td>
                            <td>${(modelData.train.accuracy * 100).toFixed(2)}%</td>
                            <td class="test-metric">${(modelData.test.accuracy * 100).toFixed(2)}%</td>
                            <td>${(modelData.test.precision * 100).toFixed(2)}%</td>
                            <td>${(modelData.test.recall * 100).toFixed(2)}%</td>
                        </tr>
                    `;
                }
            });

            html += `
                    </tbody>
                </table>
            </div>
        `;

        // Adiciona o gráfico de comparação se disponível
        if (results.comparison_chart) {
            html += `
                <div class="comparison-chart-container">
                    <h4>Gráfico Comparativo de Acurácia</h4>
                    <img src="data:image/png;base64,${results.comparison_chart}" 
                         alt="Comparação de Modelos" class="comparison-chart-img">
                </div>
            `;
        }

        comparisonResults.innerHTML = html;
    }

    // Função para buscar informações de feature importance
    function getFeatureImportance() {
        fetch('/feature_importance')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Falha ao obter importância das características');
                }
                return response.json();
            })
            .then(data => {
                if (data.chart) {
                    if (comparisonResults && !comparisonResults.classList.contains('hidden')) {
                        // Adiciona o gráfico à seção de comparação se estiver visível
                        comparisonResults.innerHTML += `
                            <div class="chart-container">
                                <h4>Importância das Características</h4>
                                <img src="data:image/png;base64,${data.chart}" 
                                     alt="Importância das Características" class="img-fluid">
                            </div>
                        `;
                    }
                }
            })
            .catch(error => {
                console.error('Erro ao obter importância das características:', error);
            });
    }

    // Helper functions
    // Variável para controlar o estado de carregamento
    let isLoading = false;
    
    function showLoading(message = "Processando...") {
        // Atualiza apenas a mensagem se já estiver carregando
        if (isLoading) {
            const loadingText = document.getElementById('loading-text');
            if (loadingText) {
                loadingText.textContent = message;
            }
            return;
        }
        
        isLoading = true;
        const loadingSpinner = document.getElementById('loading-spinner');
        const loadingText = document.getElementById('loading-text');
        
        if (loadingSpinner) {
            // Define a mensagem de carregamento
            if (loadingText) {
                loadingText.textContent = message;
            }
            
            // Exibe o spinner com animação suave
            loadingSpinner.style.opacity = '0';
            loadingSpinner.classList.remove('hidden');
            
            // Anima a entrada do spinner
            setTimeout(() => {
                loadingSpinner.style.opacity = '1';
            }, 10);
        }
    }

    function hideLoading() {
        if (!isLoading) return;
        
        const loadingSpinner = document.getElementById('loading-spinner');
        if (loadingSpinner) {
            // Anima a saída do spinner
            loadingSpinner.style.opacity = '0';
            
            // Espera a animação terminar antes de esconder completamente
            setTimeout(() => {
                loadingSpinner.classList.add('hidden');
                isLoading = false;
            }, 300);
        } else {
            isLoading = false;
        }
    }

    // Hide spinner initially
    hideLoading();

    // Attach event listeners to buttons
    if (btnTrainKnn) btnTrainKnn.addEventListener('click', () => trainModel('knn'));
    if (btnTrainDt) btnTrainDt.addEventListener('click', () => trainModel('decision_tree'));
    if (btnTrainRf) btnTrainRf.addEventListener('click', () => trainModel('random_forest'));
    if (btnTrainSvm) btnTrainSvm.addEventListener('click', () => trainModel('svm'));
    if (btnTrainAll) btnTrainAll.addEventListener('click', () => trainModel('all'));

    if (btnTestKnn) btnTestKnn.addEventListener('click', () => testModel('knn'));
    if (btnTestDt) btnTestDt.addEventListener('click', () => testModel('decision_tree'));
    if (btnTestRf) btnTestRf.addEventListener('click', () => testModel('random_forest'));
    if (btnTestSvm) btnTestSvm.addEventListener('click', () => testModel('svm'));

    if (btnPredict) btnPredict.addEventListener('click', predict);
    if (btnCompareAll) btnCompareAll.addEventListener('click', compareModels);
});