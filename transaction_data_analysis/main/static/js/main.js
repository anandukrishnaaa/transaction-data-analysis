document.getElementById('downloadPdfButton').addEventListener('click', function () {
    this.disabled = true;
    // Enable the button after 30 seconds
    setTimeout(() => {
        this.disabled = false;
    }, 30000);
    const reportId = this.getAttribute('data-file-id');
    // Use html2pdf to generate and download the PDF
    const chartsDiv = document.getElementById('charts');
    if (chartsDiv) {
        const pdfOptions = {
            margin: 10,  // Optional: Set margin (in mm)
            filename: `Report for ${reportId}.pdf`,  // Optional: Set filename
            image: { type: 'jpeg', quality: 0.98 },  // Optional: Image settings
            html2canvas: { scale: 2 },  // Optional: html2canvas settings
            jsPDF: { unit: 'mm', format: 'a3', orientation: 'landscape' }  // Set page orientation and size
        };

        html2pdf(chartsDiv, pdfOptions);
    } else {
        console.error('Element with id "charts" not found.');
    }
});

function goBack() {
    window.history.back();
}