import { ConfigProvider, App as AntdApp } from 'antd';
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
      <AntdApp>
        <ChatBot />
      </AntdApp>
    </ConfigProvider>
  );
};

export default App;
