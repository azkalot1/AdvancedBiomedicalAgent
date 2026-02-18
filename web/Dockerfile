FROM node:20-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# Build the app
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build args for env vars needed at build time
ARG NEXT_PUBLIC_BIOAGENT_ASSISTANT_ID=co_scientist
ENV NEXT_PUBLIC_BIOAGENT_ASSISTANT_ID=$NEXT_PUBLIC_BIOAGENT_ASSISTANT_ID

RUN npm run build

# Production runner
FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

# Copy built app
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]