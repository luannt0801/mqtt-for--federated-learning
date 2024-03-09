### MQTT connect

switch to switch port

access port

có arp đi nhưng ko có gói tin về
arp broadcast

về qua 2 switch nên ko tìm đc máy đích

### Cách sử dụng QoS:
Sử dụng QoS 0 khi:
Bạn có kết nối hoàn toàn hoặc gần như ổn định giữa người gửi và người nhận. Trường hợp sử dụng cổ điển cho QoS 0 là kết nối máy khách thử nghiệm hoặc ứng dụng giao diện người dùng với nhà môi giới MQTT qua kết nối có dây.

Bạn không phiền nếu thỉnh thoảng một vài tin nhắn bị mất. Việc mất một số tin nhắn có thể được chấp nhận nếu dữ liệu không quan trọng hoặc khi dữ liệu được gửi trong khoảng thời gian ngắn

Bạn không cần xếp hàng tin nhắn. Tin nhắn chỉ được xếp hàng đợi đối với các máy khách bị ngắt kết nối nếu chúng có QoS 1 hoặc 2 và phiên liên tục .

Sử dụng QoS 1 khi:
Bạn cần nhận được mọi tin nhắn và trường hợp sử dụng của bạn có thể xử lý các bản sao. QoS cấp 1 là cấp dịch vụ được sử dụng thường xuyên nhất vì nó đảm bảo tin nhắn đến ít nhất một lần nhưng cho phép gửi nhiều lần. Tất nhiên, ứng dụng của bạn phải chấp nhận sự trùng lặp và có thể xử lý chúng cho phù hợp.

Bạn không thể chịu được chi phí của QoS 2. QoS 1 gửi tin nhắn nhanh hơn nhiều so với QoS 2.

Sử dụng QoS 2 khi:
Điều quan trọng đối với ứng dụng của bạn là nhận được tất cả tin nhắn chính xác một lần. Điều này thường xảy ra nếu việc phân phối trùng lặp có thể gây hại cho người dùng ứng dụng hoặc khách hàng đăng ký. Hãy lưu ý đến chi phí chung và tương tác QoS 2 sẽ mất nhiều thời gian hơn để hoàn thành.

Xếp hàng tin nhắn QoS 1 và 2
Tất cả các tin nhắn được gửi với QoS 1 và 2 sẽ được xếp hàng đợi cho các máy khách ngoại tuyến cho đến khi máy khách khả dụng trở lại. Tuy nhiên, việc xếp hàng này chỉ có thể thực hiện được nếu máy khách có một phiên liên tục .

1. Client đăng kí tham gia FL-CSS: Server gửi client_id để xác định client
								   Server tạo topic cho client đó: Client_id
								   
2. 