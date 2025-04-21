import router from "./routes/Router.js";
import { RouterProvider } from 'react-router-dom';

function App() {
  return <RouterProvider router={router} />;
}

export default App;
