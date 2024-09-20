let charts = {};

document.addEventListener('DOMContentLoaded', function() {
    const ctx1 = document.getElementById('myChart1').getContext('2d');
    const ctx2 = document.getElementById('myChart2').getContext('2d');
    const ctx3 = document.getElementById('myChart3').getContext('2d');
    const ctx4 = document.getElementById('myChart4').getContext('2d');

    function createConfig(chartLabel) {
        return {
            type: 'line',
            data: {
                labels: [], // 초기에는 빈 배열로 설정
                datasets: [{
                    label: chartLabel,
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        };
    }

    charts.myChart1 = new Chart(ctx1, createConfig('Chart 1'));
    charts.myChart2 = new Chart(ctx2, createConfig('Chart 2'));
    charts.myChart3 = new Chart(ctx3, createConfig('Chart 3'));
    charts.myChart4 = new Chart(ctx4, createConfig('Chart 4'));
});

function changeData(datasetName) {
    const dataset = getDataset(datasetName);
    if (dataset) {
        // 레이블 업데이트
        const labels = dataset.labels || [];

        for (let chartId in charts) {
            if (dataset[chartId]) {
                charts[chartId].data.labels = labels;
                charts[chartId].data.datasets[0].data = dataset[chartId];
                charts[chartId].update();
            }
        }
    }
}

function getDataset(datasetName) {
    const datasets = {
        'dataset1': {
            labels: ['1월', '2월', '3월', '4월', '5월', '6월'],
            myChart1: [10, 20, 30, 40, 50, 60],
            myChart2: [15, 25, 35, 45, 55, 65],
            myChart3: [5, 15, 25, 35, 45, 55],
            myChart4: [20, 30, 40, 50, 60, 70]
        },
        'dataset2': {
            labels: ['1월', '2월', '3월', '4월', '5월', '6월'],
            myChart1: [60, 50, 40, 30, 20, 10],
            myChart2: [55, 45, 35, 25, 15, 5],
            myChart3: [65, 55, 45, 35, 25, 15],
            myChart4: [50, 40, 30, 20, 10, 0]
        },
        'dataset3': {
            labels: ['A', 'B', 'C'],
            myChart1: [35, 25, 45],
            myChart2: [40, 30, 50],
            myChart3: [30, 20, 40],
            myChart4: [45, 35, 55]
        },
        'dataset4': {
            labels: ['X', 'Y'],
            myChart1: [5, 15],
            myChart2: [10, 20],
            myChart3: [0, 10],
            myChart4: [15, 25]
        }
    };

    if (!datasets[datasetName]) {
        console.error('잘못된 데이터셋 이름:', datasetName);
        return null;
    }

    return datasets[datasetName];
}
