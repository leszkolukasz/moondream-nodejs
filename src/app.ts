import { Moondream } from "./index";

const moondram = await Moondream.load("../moondream-mobile/assets/models"); // Adjust the path as needed
const res = await moondram.caption("frieren.jpg", "normal", 50);
console.log(res);
