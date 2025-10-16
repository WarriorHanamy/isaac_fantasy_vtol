![Isaac Drone Racer](media/motion_trace1.jpg)

---

# Setup
In your local docker config.json, please setup your contrainer proxy (build/run process)
typically in host `~/.docker/config.json`, e.g.:
```json
{
 "proxies":
 {
   "default":
   {
     "httpProxy": "http://127.0.1:7897",
     "httpsProxy": "http://127.0.0.1:7897",
     "noProxy": "localhost,127.0.0.1,.daocloud.io"
   }
 }
}
```
