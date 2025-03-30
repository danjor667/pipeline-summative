// main.js - Main JavaScript file for Diabetes Risk Predictor

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initTooltips();

    // Initialize form validation
    initFormValidation();

    // Add smooth scrolling to all links
    initSmoothScroll();

    // Handle prediction form if it exists
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        setupPredictionForm(predictionForm);
    }

    // Setup any charts if they exist on the page
    setupCharts();

    // Setup profile tabs if they exist
    setupProfileTabs();
});

// Initialize Bootstrap tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize form validation
function initFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');

    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }

            form.classList.add('was-validated');
        }, false);
    });
}

// Handle smooth scrolling for anchor links
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Handle prediction form
function setupPredictionForm(form) {
    // Add range input value display
    const rangeInputs = form.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        const valueDisplay = document.createElement('span');
        valueDisplay.classList.add('ms-2', 'badge', 'bg-primary');
        valueDisplay.textContent = input.value;

        input.parentNode.appendChild(valueDisplay);

        input.addEventListener('input', () => {
            valueDisplay.textContent = input.value;
        });
    });

    // Add BMI calculator if height and weight fields exist
    const heightInput = form.querySelector('#id_height');
    const weightInput = form.querySelector('#id_weight');
    const bmiDisplay = document.querySelector('#bmi-value');
    const bmiInput = document.getElementById("bmi-input");


    if (heightInput && weightInput && bmiDisplay) {
        const calculateBMI = () => {
            const height = parseFloat(heightInput.value) / 100; // Convert cm to m
            const weight = parseFloat(weightInput.value);

            if (height > 0 && weight > 0) {
                const bmi = (weight / (height * height)).toFixed(1);
                bmiDisplay.textContent = bmi;
                bmiInput.value = bmi;

                // Update BMI classification
                const bmiClass = document.querySelector('#bmi-classification');
                if (bmiClass) {
                    if (bmi < 18.5) {
                        bmiClass.textContent = 'Underweight';
                        bmiClass.className = 'text-info';
                    } else if (bmi < 25) {
                        bmiClass.textContent = 'Normal';
                        bmiClass.className = 'text-success';
                    } else if (bmi < 30) {
                        bmiClass.textContent = 'Overweight';
                        bmiClass.className = 'text-warning';
                    } else {
                        bmiClass.textContent = 'Obese';
                        bmiClass.className = 'text-danger';
                    }
                }
            }
        };

        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);
    }

    // Handle form submission with AJAX if needed
    // form.addEventListener('submit', function(e) {
    //     const submitType = form.getAttribute('data-submit-type');
    //     if (submitType === 'ajax') {
    //         e.preventDefault();
    //
    //         const formData = new FormData(form);
    //         const resultContainer = document.querySelector('#prediction-result');
    //
    //         if (resultContainer) {
    //             resultContainer.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Analyzing your data...</p></div>';
    //
    //             fetch(form.action, {
    //                 method: 'POST',
    //                 body: formData,
    //                 headers: {
    //                     'X-Requested-With': 'XMLHttpRequest'
    //                 }
    //             })
    //             .then(response => response.json())
    //             .then(data => {
    //                 resultContainer.innerHTML = data.html;
    //
    //                 // Animate the risk indicator position
    //                 const riskIndicator = resultContainer.querySelector('.risk-indicator');
    //                 if (riskIndicator) {
    //                     setTimeout(() => {
    //                         riskIndicator.style.left = data.risk_percentage + '%';
    //                     }, 100);
    //                 }
    //             })
    //             .catch(error => {
    //                 resultContainer.innerHTML = '<div class="alert alert-danger">An error occurred while processing your request. Please try again.</div>';
    //                 console.error('Error:', error);
    //             });
    //         }
    //     }
    // });
}

// Setup data visualization charts
function setupCharts() {
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') return;

    // Risk Factors Chart
    const riskFactorsChart = document.getElementById('risk-factors-chart');
    if (riskFactorsChart) {
        const ctx = riskFactorsChart.getContext('2d');
        const data = JSON.parse(riskFactorsChart.getAttribute('data-values'));
        const labels = JSON.parse(riskFactorsChart.getAttribute('data-labels'));

        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Your Risk Factors',
                    data: data,
                    backgroundColor: 'rgba(13, 110, 253, 0.2)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(13, 110, 253, 1)',
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 10
                    }
                }
            }
        });
    }

    // Progress Chart
    const progressChart = document.getElementById('progress-chart');
    if (progressChart) {
        const ctx = progressChart.getContext('2d');
        const data = JSON.parse(progressChart.getAttribute('data-values'));
        const labels = JSON.parse(progressChart.getAttribute('data-labels'));

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Risk Score Over Time',
                    data: data,
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Risk Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });
    }
}

// Setup profile tabs
function setupProfileTabs() {
    const tabLinks = document.querySelectorAll('.profile-tab-link');
    if (tabLinks.length === 0) return;

    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            // Remove active class from all tabs
            tabLinks.forEach(l => l.classList.remove('active'));

            // Add active class to clicked tab
            this.classList.add('active');

            // Show corresponding content
            const target = this.getAttribute('data-target');
            document.querySelectorAll('.profile-tab-content').forEach(content => {
                content.style.display = 'none';
            });

            document.querySelector(target).style.display = 'block';

            // Update URL hash
            history.pushState(null, null, this.getAttribute('href'));
        });
    });

    // Check if there's a hash in the URL and activate that tab
    if (window.location.hash) {
        const activeTab = document.querySelector(`.profile-tab-link[href="${window.location.hash}"]`);
        if (activeTab) {
            activeTab.click();
        }
    }
}

// Function to handle the risk prediction results animation
function animateRiskMeter(percentage) {
    const riskIndicator = document.querySelector('.risk-indicator');
    if (riskIndicator) {
        riskIndicator.style.left = percentage + '%';
    }
}

// Print results function
function printResults() {
    window.print();
}

// Save results to account (if logged in)
function saveResults(resultId) {
    const saveButton = document.getElementById('save-result-btn');
    if (!saveButton) return;

    saveButton.disabled = true;
    saveButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';

    fetch('/save-result/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({ result_id: resultId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            saveButton.innerHTML = '<i class="bi bi-check-circle"></i> Saved';
            saveButton.classList.remove('btn-primary');
            saveButton.classList.add('btn-success');
        } else {
            saveButton.innerHTML = 'Save Result';
            saveButton.disabled = false;
            alert('Error saving result. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        saveButton.innerHTML = 'Save Result';
        saveButton.disabled = false;
        alert('An error occurred. Please try again.');
    });
}

// Helper function to get CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}