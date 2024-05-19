import React, { useRef, useState } from "react";
import "./App.css";
import Heading from "./components/Heading";
// import Webcam from "react-webcam";
import { AiOutlineCloudUpload } from "react-icons/ai";
import Describe from "./components/Describe";
import FootCont from "./components/FootCont";
import img1 from "../public/output.png"
function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [backendFiles, setBackendFiles] = useState([]); // State to hold files from the backend
  const [images, setImages] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const files = event.target.files;
    const newFiles = [];
    for (let i = 0; i < files.length; i++) {
      newFiles.push(files[i]);
      if (files[i].type.split('/')[0] === 'image' && !images.some((e) => e.name === files[i].name)) {
        setImages((prevImages) => [
          ...prevImages,
          {
            name: files[i].name,
            url: URL.createObjectURL(files[i]),
          },
        ]);
      }
    }
    setSelectedFiles((prevFiles) => [...prevFiles, ...newFiles]);
  };

  const sendFileToBackend = async () => {
    if (selectedFiles.length === 0) {
      alert("No files selected");
      return;
    }

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("file", file); // Use "file" instead of "files"
    });

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const files = await response.json();
        setBackendFiles(files); // Assume the backend returns an array of file URLs
      } else {
        console.error("File upload failed");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  function selectFiles() {
    fileInputRef.current.click();
  }

  function onFileSelect(event) {
    handleFileChange(event);
  }

  function deleteImage(index) {
    setImages((prevImages) => prevImages.filter((_, i) => i !== index));
    setSelectedFiles((prevFiles) => prevFiles.filter((_, i) => i !== index));
  }

  function onDragOver(event) {
    event.preventDefault();
    setIsDragging(true);
    event.dataTransfer.dropEffect = 'copy';
  }

  function onDragLeave(event) {
    event.preventDefault();
    setIsDragging(false);
  }

  function onDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    const files = event.dataTransfer.files;
    handleFileChange({ target: { files } });
  }

  return (
    <>
      <div className="head_container">
        <Heading />
      </div>
      <Describe></Describe>
      {/* <div className="web_container"><Webcam ref={webRef}/></div> */}
      <div className="main-container">
        <div className="card">
          <div className="top">
            <p>Drag & Drop image uploading</p>
          </div>
          <div className="drag-area" onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
            {isDragging ? (
              <span className="select">Drop images here</span>
            ) : (
              <>
                Drag & Drop images here or{" "}
                <span className="select" role="button" onClick={selectFiles}><AiOutlineCloudUpload /></span>
              </>
            )}
            <input
              type="file"
              className="file"
              multiple
              ref={fileInputRef}
              onChange={onFileSelect}
              style={{ display: 'none' }}
            />
          </div>
          <div className="container">
            {images.map((image, index) => (
              <div className="image" key={index}>
                <span className="delete" onClick={() => deleteImage(index)}>&times;</span>
                <img src={image.url} alt={image.name} />
              </div>
            ))}
          </div>
          <button type="button" onClick={sendFileToBackend}>Upload</button>
        </div>

        <div className="uploaded-files">
          {backendFiles.map((file, index) => (
            file.type.split('/')[0] === 'image' ? (
              <img key={index} src={img1} alt={`Uploaded ${file.name}`} />
               
          ) : (
          <video key={index} controls>
            <source src={file.url} type={file.type} />
            Your browser does not support the video tag.
          </video>
          )
        ))}
       
        </div>
      </div>
      <div className="footer">
        <hr />
        <FootCont></FootCont>
      </div>
    </>
  );
}

export default App;
