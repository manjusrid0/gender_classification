navigator.mediaDevices.getUserMedia({video:true})
.then(stream=>{
document.getElementById("video").srcObject=stream
})

function capture(){

const video=document.getElementById("video")
const canvas=document.getElementById("canvas")

const ctx=canvas.getContext("2d")

ctx.drawImage(video,0,0,280,280)

const image=canvas.toDataURL("image/png")

fetch("/camera_predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({image:image})

})

.then(res=>res.json())

.then(data=>{

document.getElementById("result").innerHTML =
"Prediction : "+data.prediction+" ("+data.confidence+"%)"

})

}