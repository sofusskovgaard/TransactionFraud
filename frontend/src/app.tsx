import react, { useEffect } from 'react'
import io from 'socket.io-client'

const App = () => {
  const socket = io("ws://localhost:5000")

  useEffect(() => {

    socket.connect()

    socket.on("hello", arg => {
      console.log(arg)
    })

    fetch('http://localhost:5000/hello-world', { method: 'GET' }).then(res => {
      res.text().then(ress => console.log('fetch response', ress))
    })
  }, [])

  function sendMsg() {
    socket.emit("hello", "world", (response) => {
      console.log("response", response);
    })
  }

  return (
    <div>
      <h1>Hello World!</h1>
      <button onClick={sendMsg}>send msg</button>
    </div>
  )
}

export default App