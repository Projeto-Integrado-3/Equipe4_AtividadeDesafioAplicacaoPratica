/**
 * Creates a bar chart for comparing model metrics
 * @param {string} elementId - The ID of the canvas element
 * @param {Array} modelNames - Array of model names
 * @param {Array} trainValues - Array of training values
 * @param {Array} testValues - Array of test values
 * @param {string} metricName - Name of the metric (e.g., "Accuracy")
 */
function createComparisonBarChart(elementId, modelNames, trainValues, testValues, metricName) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: `${metricName} (Treino)`,
                    data: trainValues,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: `${metricName} (Teste)`,
                    data: testValues,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
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
            },
            plugins: {
                title: {
                    display: true,
                    text: `Comparação de ${metricName}`
                }
            }
        }
    });
}

/**
 * Creates a radar chart for comparing multiple metrics across models
 * @param {string} elementId - The ID of the canvas element
 * @param {Array} models - Array of model data objects
 */
function createMetricsRadarChart(elementId, models) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return;

    const metrics = ['accuracy', 'precision', 'recall'];
    const labels = metrics.map(m => m.charAt(0).toUpperCase() + m.slice(1));

    const datasets = models.map((model, index) => {
        const colors = [
            'rgba(54, 162, 235, 0.7)',  // blue
            'rgba(255, 99, 132, 0.7)',  // red
            'rgba(75, 192, 192, 0.7)',  // green
            'rgba(153, 102, 255, 0.7)'  // purple
        ];

        return {
            label: model.name,
            data: metrics.map(metric => model.test[metric]),
            backgroundColor: colors[index % colors.length].replace('0.7', '0.2'),
            borderColor: colors[index % colors.length],
            pointBackgroundColor: colors[index % colors.length],
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: colors[index % colors.length]
        };
    });

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}

/**
 * Creates a doughnut chart for showing class distribution
 * @param {string} elementId - The ID of the canvas element
 * @param {Array} classNames - Array of class names
 * @param {Array} counts - Array of sample counts per class
 */
function createClassDistributionChart(elementId, classNames, counts) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: classNames,
            datasets: [{
                data: counts,
                backgroundColor: [
                    'rgba(61, 153, 112, 0.7)',  // verde
                    'rgba(74, 144, 226, 0.7)',  // azul
                    'rgba(106, 27, 154, 0.7)'   // roxo
                ],
                borderColor: [
                    'rgba(61, 153, 112, 1)',
                    'rgba(74, 144, 226, 1)',
                    'rgba(106, 27, 154, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'Distribuição de Classes'
                }
            }
        }
    });
}

/**
 * Creates a bar chart for showing feature importance
 * @param {string} elementId - The ID of the canvas element
 * @param {Array} featureNames - Array of feature names
 * @param {Array} importanceValues - Array of importance values
 */
function createFeatureImportanceChart(elementId, featureNames, importanceValues) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return;

    // Sort features by importance
    const combined = featureNames.map((name, i) => ({ name, value: importanceValues[i] }));
    combined.sort((a, b) => b.value - a.value);
    
    const sortedNames = combined.map(item => item.name);
    const sortedValues = combined.map(item => item.value);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedNames,
            datasets: [{
                label: 'Importância',
                data: sortedValues,
                backgroundColor: 'rgba(75, 192, 192, 0.7)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Importância das Características'
                }
            }
        }
    });
}

/**
 * Creates a bar chart for showing prediction probabilities
 * @param {string} elementId - The ID of the canvas element
 * @param {Object} probabilities - Object with class names as keys and probabilities as values
 */
function createProbabilityChart(elementId, probabilities) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return;

    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities);

    // Define cores para cada classe
    const backgroundColors = [
        'rgba(61, 153, 112, 0.7)',  // setosa
        'rgba(74, 144, 226, 0.7)',  // versicolor
        'rgba(106, 27, 154, 0.7)'   // virginica
    ];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probabilidade',
                data: values,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
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
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Probabilidades por Classe'
                }
            }
        }
    });
}