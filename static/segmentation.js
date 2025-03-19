// Get elements
const uploadForm = document.getElementById('upload-form');
const imageUpload = document.getElementById("image-upload");
const previewImage = document.getElementById("preview-image");
const uploadPrompt = document.querySelector(".upload-prompt");
const processBtn = document.getElementById("process-btn");
const loadingSpinner = document.getElementById('loading-spinner');
loadingSpinner.style.display = 'none'
const resultContainer = document.getElementById("result-container");
const originalResult = document.getElementById("original-result");
const segmentedResult = document.getElementById("segmented-result");

// Handle image upload and preview
imageUpload.addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    const reader = new FileReader();

    reader.onload = function (e) {
      previewImage.src = e.target.result;
      previewImage.style.display = "block";
      uploadPrompt.style.display = "none";
      processBtn.disabled = false;
    };

    reader.readAsDataURL(file);
  }
});

// Form submission handler
uploadForm.addEventListener('submit',function(e){
  e.preventDefault();

  loadingSpinner.style.display = 'block';
  processBtn.disabled = true;

  const formData = new FormData(this);

  fetch('/segment',{
      method: 'POST',
      body: formData
  })
  .then(response =>{
      if (!response.ok){
          throw new Error('Network response was not ok')
      }
      return response.json();
  })
  .then(data => {
          // Hide loading spinner
          loadingSpinner.style.display = 'none';
          processBtn.disabled = false;
          
          // Show result container
          resultContainer.style.display = 'block';
          
          // Set original image
          originalResult.src = data.original_image;
          
          // Set segmented image
          segmentedResult.src = data.segmented_image;
                          
          // Scroll to results
          resultContainer.scrollIntoView({ behavior: 'smooth' });
      })
      .catch(error => {
          console.error('Error:', error);
          loadingSpinner.style.display = 'none';
          processBtn.disabled = false;
          alert('Error processing image. Please try again.');
      });
  });
  
// Allow drag and drop on the preview container
const previewContainer = document.querySelector(".preview-container");

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  previewContainer.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

["dragenter", "dragover"].forEach((eventName) => {
  previewContainer.addEventListener(eventName, highlight, false);
});

["dragleave", "drop"].forEach((eventName) => {
  previewContainer.addEventListener(eventName, unhighlight, false);
});

function highlight() {
  previewContainer.style.borderColor = "#198754";
  previewContainer.style.backgroundColor = "rgba(25, 135, 84, 0.1)";
}

function unhighlight() {
  previewContainer.style.borderColor = "#ccc";
  previewContainer.style.backgroundColor = "";
}

previewContainer.addEventListener("drop", handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;

  if (files.length) {
    imageUpload.files = files;
    // Trigger the change event manually
    const event = new Event("change");
    imageUpload.dispatchEvent(event);
  }
}
