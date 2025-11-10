import { ConfigProvider } from 'antd';
import ChatBot from './pages/ChatBot';
import './App.css';

const App = () => {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#005489',
          borderRadius: 8,
          fontSize: 14,
        },
      }}
    >
      <ChatBot />
    </ConfigProvider>
  );
};

export default App;
