// src/components/ImageCarousel.js
import React, { useState, useEffect } from 'react';
import { Carousel } from 'antd';
import './ImageCarousel.css';

const ImageCarousel = () => {
  const [images] = useState([
    '/technical-drawing-1.jpg',
    '/harness-design.jpg',
    '/vehicle-layout.png',
    '/component-diagram.svg'
  ]);

  return (
    <div className="corner-carousel top-right">
      <Carousel autoplay dotPosition="top">
        {images.map((img, index) => (
          <div key={index}>
            <img 
              src={img} 
              alt={`Technical illustration ${index + 1}`} 
              className="carousel-image"
            />
          </div>
        ))}
      </Carousel>
    </div>
  );
};

export default ImageCarousel;