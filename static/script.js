
const mixedImages = [
  "https://dynamic.realestateindia.com/prop_images/3649219/1317300_1-350x350.jpg",
  "https://mediacdn.99acres.com/1588/16/31776167O-1427547187.jpeg",
  "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRnKcc9-OD6zd0iR0r5C9wqwWadbMLwRVEsgQ&s",
  "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwz7cf22HGjLHPGpSggSHioilr6GJ01JL0Ww&s"
];

const images=[
  "https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=1400&q=60", 
  "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1400&q=60", 
  "https://images.unsplash.com/photo-1599423300746-b62533397364?auto=format&fit=crop&w=1400&q=60", 
  "https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=1400&q=60"
];

const sampleListings = [
  { title: "Modern 3BHK", area: "Mahalakshmi Layout", beds: 3, baths: 2, sqft: 1350, price: 9500000, img: mixedImages[0] },
  { title: "Luxury Villa with Pool", area: "Indhranagar", beds: 4, baths: 4, sqft: 4200, price: 12500000, img: mixedImages[1] },
  { title: "1BHK Flat", area: "Maruthinagar", beds: 1, baths: 1, sqft: 850, price: 2500000, img: mixedImages[2] },
  { title: "Farm house", area: "Bommasandhra", beds: 4, baths: 3, sqft: 2100, price: 18500000, img: mixedImages[3] }
];

/* Helpers */
function fmtINR(num){
  if(num===null || num===undefined) return "—";
  return "₹ " + Number(num).toLocaleString("en-IN");
}
function setYear(ids){
  const y = new Date().getFullYear();
  ids.forEach(id=>{ const el=document.getElementById(id); if(el) el.innerText=y; });
}

/* HERO SLIDER */
function initHeroSlider(){
  const container = document.getElementById("hero-slider");
  if(!container) return;
  images.forEach((src,i)=>{
    const img=document.createElement("img");
    img.src=src; img.alt="House image "+(i+1);
    if(i!==0) img.style.display="none";
    container.appendChild(img);
  });
  let index=0;
  setInterval(()=>{
    const imgs=container.querySelectorAll("img");
    imgs.forEach((im,i)=> im.style.display = (i===index) ? "block" : "none");
    index=(index+1)%imgs.length;
  },3500);
}

/* LISTINGS */
function renderListings(listings=sampleListings){
  const grid=document.getElementById("listings-grid"); if(!grid) return;
  grid.innerHTML="";
  listings.forEach(l=>{
    const card=document.createElement("div"); card.className="card";
    card.innerHTML=`
      <div class="thumb"><img src="${l.img}" alt="${l.title}" /></div>
      <div class="body">
        <div class="title">${l.title}</div>
        <div class="meta">${l.area} • ${l.beds} beds • ${l.baths} baths • ${l.sqft} sqft</div>
        <div class="price">${fmtINR(l.price)}</div>
      </div>`;
    grid.appendChild(card);
  });
}

/* SEARCH */
function initSearch(){
  const btn=document.getElementById("search-btn"); if(!btn) return;
  btn.addEventListener("click",()=>{
    const city=document.getElementById("search-city").value.trim().toLowerCase();
    const type=document.getElementById("search-type").value;
    const filtered = sampleListings.filter(l=>{
      const matchesCity = city ? l.area.toLowerCase().includes(city) : true;
      const matchesType = type ? (
        (type==="villa" && l.title.toLowerCase().includes("villa")) ||
        (type==="apartment" && l.title.toLowerCase().includes("apartment")) ||
        (type==="house" && l.title.toLowerCase().includes("house"))
      ) : true;
      return matchesCity && matchesType;
    });
    renderListings(filtered);
  });
}

/* PREDICT PAGE */
function initPredictPage(){
  const btn = document.getElementById("predict-btn");
  if (!btn) return;

  // carousel setup
  const carousel = document.getElementById("carousel-images");
  if (carousel) {
    images.forEach(src => {
      const im = document.createElement("img");
      im.src = src;
      im.alt = "property photo";
      im.style.display = "none";
      carousel.appendChild(im);
    });
    const imgs = carousel.querySelectorAll("img");
    if (imgs[0]) imgs[0].style.display = "block";
    let cur = 0;

    const next = document.getElementById("carousel-next"),
          prev = document.getElementById("carousel-prev");

    if (next)
      next.addEventListener("click", () => {
        imgs[cur].style.display = "none";
        cur = (cur + 1) % imgs.length;
        imgs[cur].style.display = "block";
      });

    if (prev)
      prev.addEventListener("click", () => {
        imgs[cur].style.display = "none";
        cur = (cur - 1 + imgs.length) % imgs.length;
        imgs[cur].style.display = "block";
      });
  }

  // ------- PREDICT BUTTON LOGIC -------
  btn.addEventListener("click", async () => {
    const location = document.getElementById("location").value.trim();
    const sqft = Number(document.getElementById("sqft").value);
    const bath = Number(document.getElementById("bath").value);
    const bhk = Number(document.getElementById("bhk").value);

    if (!location || !sqft || !bath || !bhk) {
             alert("Please fill location, sqft, bath, and bhk.");
             return;
}


    const loading = document.getElementById("predict-loading");
    const priceValue = document.getElementById("price-value");
    const simPrice = document.getElementById("sim-price");

    loading.classList.remove("hidden");
    priceValue.innerText = "—";
    simPrice.innerText = "—";

    try {
      const payload = {
      location: location,
      total_sqft: sqft,
      bath: bath,
      bhk: bhk
};


      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      const p = data.estimated_price_lakhs;

      if (data.error) {
                 priceValue.innerText = "Error: " + data.error;
                 console.error("Backend error:", data.error);
                loading.classList.add("hidden");
               return;
}
        priceValue.innerText = p + " Lakhs";



      // similar listing by nearest sqft
      const similar = sampleListings.reduce((acc, cur) => {
        const diff = Math.abs(cur.sqft - sqft);
        return diff < acc.diff ? { price: cur.price, diff } : acc;
      }, { price: null, diff: Infinity });

      simPrice.innerText = similar.price
        ? "₹ " + Number(similar.price).toLocaleString("en-IN")
        : "—";

    } catch (err) {
      console.error(err);
      alert("Backend error. Ensure Flask backend is running.");
    } finally {
      loading.classList.add("hidden");
    }
  });
}

/* CONTACT */
function initContact(){
  const btn = document.getElementById("contact-send");
  if (!btn) return;

  btn.addEventListener("click", async () => {
document.getElementById("contact-send").addEventListener("click", async () => {
  const name = document.getElementById("contact-name").value.trim();
  const email = document.getElementById("contact-email").value.trim();
  const message = document.getElementById("contact-message").value.trim();
  const feedback = document.getElementById("contact-feedback");

  // Email regex validation
  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  if (!name || !email || !message) {
    feedback.innerHTML = "⚠️ Please fill all fields.";
    feedback.style.color = "red";
    return;
  }

  if (!emailPattern.test(email)) {
    feedback.innerHTML = "⚠️ Please enter a valid email address.";
    feedback.style.color = "red";
    return;
  }

  feedback.innerHTML = "Sending...";
  feedback.style.color = "black";

  try {
    const response = await fetch("/send-contact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, message }),
    });

    const result = await response.json();

    if (result.success) {
      feedback.innerHTML = "✅ Message sent successfully!";
      feedback.style.color = "green";

      // Clear fields
      document.getElementById("contact-name").value = "";
      document.getElementById("contact-email").value = "";
      document.getElementById("contact-message").value = "";
    } else {
      feedback.innerHTML = "❌ Failed to send message.";
      feedback.style.color = "red";
    }
  } catch (error) {
    feedback.innerHTML = "❌ Server error. Try again later.";
    feedback.style.color = "red";
  }
});

});
}
/* INIT on DOM ready */
document.addEventListener("DOMContentLoaded", ()=>{
  setYear(["year","year2","year3","year4"]);
  initHeroSlider();
  renderListings();
  initSearch();
  initPredictPage();
  initContact();
});
