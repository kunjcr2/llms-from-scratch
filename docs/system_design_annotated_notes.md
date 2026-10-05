# System Design Annotated Notes

Source: transcript pasted in the Codex attachment.

## Transcript Corrections

- `ACT` should be `ACID`.
- `PostgresSQL` should be `PostgreSQL`.
- `SQL light` should be `SQLite`.
- `Radius` should be `Redis`.
- `Neo 4G` should be `Neo4j`.
- `memcache` is usually `Memcached`.
- `NX` should be `Nginx`.
- `AJ Proxy` should be `HAProxy`.
- `course` should be `CORS`.
- `cross-sight` should be `cross-site`.
- `OF2` or `Oaf 2` should be `OAuth 2.0`.
- `GVT` should be `JWT`.
- `SL` should be `SAML`.
- `Versel` should be `Vercel`.

## 1. Senior Engineer Mindset

- The main shift from mid-level to senior is from implementation to decision-making.
- Senior engineers are expected to handle unclear requirements, make tradeoffs, and design systems from first principles.
- Strong system design means balancing:
  - performance
  - reliability
  - scalability
  - security
  - maintainability
  - developer experience
- Note: "senior" is not just about designing big systems. It is also about making small systems easy to evolve.

## 2. Single Server Setup

- Start with one server before discussing scale.
- In the simplest setup, one machine can host:
  - web app
  - API
  - database
  - cache
- Basic request flow:
  1. user enters domain
  2. DNS resolves domain to IP
  3. client sends HTTP request
  4. server processes request
  5. server returns HTML or JSON
- Web clients usually consume HTML, CSS, and JS.
- Mobile clients usually consume JSON APIs over HTTP.
- Note: a single-server setup is useful for understanding system boundaries, not as a long-term default for growth.

## 3. Database Selection

### SQL / relational databases

- Good when data is structured and relationships matter.
- Strong fit for joins, constraints, and transactions.
- Core strength: ACID guarantees.
  - Atomicity
  - Consistency
  - Isolation
  - Durability
- Common examples: PostgreSQL, MySQL, Oracle, SQLite.

### NoSQL databases

- Good when you need flexible schemas, very large scale, or specialized access patterns.
- Main categories mentioned in the transcript:
  - document stores: MongoDB
  - wide-column stores: Cassandra
  - key-value stores: Redis, Memcached
  - graph databases: Neo4j, Amazon Neptune

### Practical notes

- Use SQL when consistency and relational integrity are the priority.
- Use NoSQL when scale, write throughput, or flexible schema matters more than relational structure.
- Note: "NoSQL is faster" is too broad. Performance depends on the query pattern, data model, indexes, and workload shape.
- Note: many production systems use both SQL and NoSQL together.

## 4. Vertical vs Horizontal Scaling

- Vertical scaling means adding more CPU, RAM, or storage to one machine.
- Horizontal scaling means adding more machines and spreading load across them.

### Vertical scaling

- Simpler operationally.
- Good for early-stage systems or moderate traffic.
- Limited by hardware ceilings.
- Still leaves a single-machine failure risk.

### Horizontal scaling

- Better for large systems.
- Improves redundancy and fault tolerance.
- Requires traffic distribution and usually stateless application design.
- Note: if the app keeps session state in memory on one server, horizontal scaling becomes harder unless session state is externalized.

## 5. Load Balancing

- A load balancer distributes requests across multiple servers.
- It improves:
  - availability
  - fault tolerance
  - scaling flexibility

### Algorithms covered

- Round robin: good when servers are similar.
- Least connections: good when request/session lengths vary.
- Least response time: good when server responsiveness differs.
- IP hash: keeps a client mapped to the same server.
- Weighted balancing: favors stronger servers.
- Geographic routing: sends traffic to a closer region.
- Consistent hashing: useful when stable routing matters and node membership changes.

### Important notes

- Health checks are critical; otherwise the load balancer cannot know which instances are unhealthy.
- A load balancer can become a single point of failure if only one exists.
- Cloud load balancers reduce operational burden but do not remove the need for good architecture.

## 6. Single Point of Failure

- A SPOF is any component whose failure can break the entire system.
- Common examples:
  - one database
  - one load balancer
  - one queue broker
  - one auth provider dependency
- Common mitigations:
  - redundancy
  - failover
  - health checks
  - self-healing / auto-replacement
  - replication
- Note: removing SPOFs often increases system complexity. That tradeoff should be intentional.

## 7. API Design Fundamentals

- An API is a contract between systems.
- It defines:
  - allowed requests
  - expected responses
  - boundaries between services
- Good API design should optimize for:
  - clarity
  - consistency
  - performance
  - evolvability

### REST vs GraphQL vs gRPC

- REST:
  - resource-oriented
  - uses standard HTTP methods
  - easiest default for most public APIs
- GraphQL:
  - client asks for exactly the fields it needs
  - good for complex UIs and nested data
  - can reduce over-fetching and under-fetching
- gRPC:
  - high-performance RPC with Protocol Buffers
  - strong fit for internal service-to-service communication
  - supports streaming well

### Protocol note

- HTTP works well for request/response APIs.
- WebSockets fit real-time bidirectional communication.
- Message queues fit asynchronous workflows.
- gRPC is strong for low-latency internal service communication.

## 8. REST API Notes

- Model resources as nouns, not verbs.
  - good: `/products`
  - bad: `/getProducts`
- Use collection and item URLs clearly.
  - `/products`
  - `/products/{id}`
  - `/products/{id}/reviews`

### Filtering, sorting, pagination

- Filtering should narrow the result set.
- Sorting should happen server-side, not in the client after fetching huge datasets.
- Pagination avoids large payloads.
- Common pagination styles:
  - page + limit
  - offset + limit
  - cursor-based pagination
- Note: cursor pagination is usually better than offset pagination for large, frequently changing datasets.

### HTTP methods

- `GET`: read
- `POST`: create
- `PUT`: replace entire resource
- `PATCH`: partial update
- `DELETE`: remove

### Status code notes

- `200 OK`: successful read or update response
- `201 Created`: successful creation
- `204 No Content`: success with no response body
- `400 Bad Request`: malformed request
- `401 Unauthorized`: unauthenticated
- `403 Forbidden`: authenticated but not allowed
- `404 Not Found`: resource missing
- `5xx`: server-side failure
- Note: the transcript mentions `401` and `404`, but `403` is also an important distinction in real APIs.

### Best practices

- Use consistent naming.
- Version intentionally.
- Return structured error payloads.
- Keep endpoint behavior predictable.

## 9. GraphQL Notes

- GraphQL solves the "too much data / too little data" problem common in REST.
- Core parts:
  - schema
  - types
  - queries
  - mutations
  - subscriptions
- Clients specify the response shape they want.

### Important notes

- Good schemas should reflect the domain model cleanly.
- Avoid deep or unbounded nesting.
- Add query depth limits and complexity limits.
- Use input types for mutations.
- Note: the transcript says GraphQL always returns `200`. That is not universally true. Many GraphQL servers use `200` for application-level errors, but transport-level failures can still return `4xx` or `5xx`.

## 10. Authentication Notes

- Authentication answers: "Who are you?"
- Authorization answers: "What are you allowed to do?"

### Methods covered

- Basic auth:
  - sends credentials every request
  - only acceptable over HTTPS
  - rarely a good modern default
- Digest auth:
  - slightly safer than basic auth
  - mostly obsolete in modern systems
- API keys:
  - simple machine-to-machine access
  - weak identity semantics unless paired with lookup, scoping, rotation, and expiration
- Session + cookie auth:
  - common for traditional server-rendered web apps
  - stateful because session data must be stored server-side
- Bearer tokens:
  - whoever holds the token can use it
  - "bearer" is a usage pattern, not a token format
- JWT:
  - signed token format
  - can carry claims like user ID, roles, and expiry

### Access and refresh tokens

- Access token:
  - short-lived
  - used for API calls
- Refresh token:
  - longer-lived
  - used to mint new access tokens
- Store refresh tokens in `HttpOnly` cookies, not local storage.
- Note: JWTs help scalability, but token revocation and immediate logout become harder than with server-side sessions.

### OAuth 2.0, OpenID Connect, SSO

- OAuth 2.0 is an authorization framework.
- OpenID Connect adds identity on top of OAuth 2.0.
- SSO is a user experience pattern, not a standalone auth method.
- SAML and OpenID Connect are identity protocols often used for SSO.

## 11. Authorization Notes

- Authorization decides what an authenticated user can access.

### Models covered

- RBAC:
  - role-based access control
  - simplest common model
  - good for coarse-grained permissions
- ABAC:
  - attribute-based access control
  - uses user, resource, and environment attributes
  - more flexible, harder to reason about
- ACL:
  - access control list attached to a specific resource
  - good for per-document sharing patterns like Google Docs

### Practical notes

- RBAC is easier to manage at scale.
- ABAC is better when policy depends on context.
- ACLs offer fine-grained sharing but can get operationally expensive at large scale.
- Tokens carry claims, but claims are not the same thing as authorization policy.

## 12. API Security Notes

- Security is layered. No single mechanism is enough.

### Techniques covered

- Rate limiting:
  - per endpoint
  - per user
  - per IP
  - global throttling for abuse spikes
- CORS:
  - browser-enforced origin policy
  - controls which frontends can call your API from browsers
- SQL / NoSQL injection prevention:
  - use parameterized queries
  - validate and sanitize inputs
- Firewalls / WAF:
  - block suspicious traffic patterns
- VPN / private network access:
  - useful for internal-only APIs
- CSRF protection:
  - especially relevant for cookie-based auth
- XSS prevention:
  - escape output
  - sanitize user-generated HTML
  - use strong content security policies where possible

### Important notes

- CORS is not authentication and not a replacement for authorization.
- CSRF matters mainly when the browser automatically sends credentials, especially cookies.
- XSS can bypass many frontend assumptions, including token theft if tokens are stored unsafely.
- Security design should include logging, monitoring, secret rotation, and incident response, even though the transcript focuses mostly on request-time defenses.

## 13. Final Takeaway

- Start simple.
- Separate concerns early.
- Choose databases and protocols based on access patterns, not trends.
- Design APIs as long-lived contracts.
- Treat auth, authz, and security as separate layers.
- Remove SPOFs before traffic growth makes failures expensive.
- The real senior-level skill is not memorizing components. It is choosing the right tradeoff for the current system and knowing what will break next.
