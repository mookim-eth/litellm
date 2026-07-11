# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:c61ac6919b811ea53c4782d69f1fe05218ba3c25d53f01b6ab7892e621bd4370

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:c61ac6919b811ea53c4782d69f1fe05218ba3c25d53f01b6ab7892e621bd4370

# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

# Set the working directory to /app
WORKDIR /app

USER root

# Install build dependencies
RUN apk add --no-cache bash gcc py3-pip python3 python3-dev openssl openssl-dev

RUN python -m pip install build==1.4.2

# Build dependency wheels before copying the full source tree.  This keeps the
# expensive dependency-resolution/download layer reusable when only LiteLLM
# Python sources change.
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements.txt

# Copy only the Admin UI inputs before the full source tree.  This allows
# Docker to reuse the UI-build layer for Python-only changes.
COPY docker/build_admin_ui.sh docker/build_admin_ui.sh
COPY enterprise/enterprise_ui/ enterprise/enterprise_ui/
COPY ui/litellm-dashboard/ ui/litellm-dashboard/

# Build Admin UI
# Convert Windows line endings to Unix and make executable
RUN sed -i 's/\r$//' docker/build_admin_ui.sh && chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Copy the current directory contents into the container at /app after cached
# dependency/UI layers are complete.
COPY . .

# Build the package
RUN rm -rf dist/* && python -m build

# There should be only one wheel file now, assume the build only creates one
RUN ls -1 dist/*.whl | head -1

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

# Ensure runtime stage runs as root
USER root

# Install runtime dependencies (libsndfile needed for audio processing on ARM64)
RUN apk add --no-cache bash openssl tzdata nodejs npm python3 py3-pip libsndfile && \
    npm install -g npm@11.12.1 tar@7.5.11 glob@11.1.0 @isaacs/brace-expansion@5.0.1 minimatch@10.2.4 diff@8.0.3 && \
    # SECURITY FIX: npm bundles tar, glob, and brace-expansion at multiple nested
    # levels inside its dependency tree. `npm install -g <pkg>` only creates a
    # SEPARATE global package, it does NOT replace npm's internal copies.
    # We must find and replace EVERY copy inside npm's directory.
    GLOBAL="$(npm root -g)" && \
    find "$GLOBAL/npm" -type d -name "tar" -path "*/node_modules/tar" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/tar" "$d"; \
    done && \
    find "$GLOBAL/npm" -type d -name "glob" -path "*/node_modules/glob" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/glob" "$d"; \
    done && \
    find "$GLOBAL/npm" -type d -name "brace-expansion" -path "*/node_modules/@isaacs/brace-expansion" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/@isaacs/brace-expansion" "$d"; \
    done && \
    find "$GLOBAL/npm" -type d -name "minimatch" -path "*/node_modules/minimatch" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/minimatch" "$d"; \
    done && \
    find "$GLOBAL/npm" -type d -name "diff" -path "*/node_modules/diff" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/diff" "$d"; \
    done && \
    # SECURITY FIX: patch npm's own package.json metadata so scanners see the
    # actual installed versions instead of the stale declared dependencies.
    find /usr/local/lib /usr/lib -path "*/node_modules/npm/package.json" -exec \
        sed -i 's/"tar": "\^7\.5\.[0-9]*"/"tar": "^7.5.10"/g; s/"minimatch": "\^10\.[0-9.]*"/"minimatch": "^10.2.4"/g' {} + 2>/dev/null && \
    npm cache clean --force && \
    # Remove the apk-tracked npm so its stale SBOM metadata (tar 7.5.9) is
    # no longer visible to image scanners.  The globally installed npm@latest
    # at /usr/local/lib/node_modules/npm/ remains fully functional.
    { apk del --no-cache npm 2>/dev/null || true; }

WORKDIR /app

# Copy and normalize runtime scripts/config that do not change with LiteLLM
# Python sources.  Keep these before the app wheel install so Python-only
# changes do not invalidate later static setup layers.
COPY docker/entrypoint.sh docker/entrypoint.sh
COPY docker/prod_entrypoint.sh docker/prod_entrypoint.sh
COPY docker/install_auto_router.sh docker/install_auto_router.sh
COPY docker/supervisord.conf /etc/supervisord.conf
COPY litellm/proxy/schema.prisma litellm/proxy/schema.prisma

COPY --from=builder /wheels/ /wheels/

# Install dependency wheels before copying the app wheel.  This is the main
# runtime cache boundary: editing LiteLLM .py files rebuilds/reinstalls only the
# application wheel instead of reinstalling all third-party dependencies.
RUN pip install /wheels/* --no-index --find-links=/wheels/ --no-deps && rm -rf /wheels

# Replace the nodejs-wheel-binaries bundled node with the system node (fixes CVE-2025-55130)
RUN NODEJS_WHEEL_NODE=$(find /usr/lib -path "*/nodejs_wheel/bin/node" 2>/dev/null) && \
    if [ -n "$NODEJS_WHEEL_NODE" ]; then cp /usr/bin/node "$NODEJS_WHEEL_NODE"; fi

# Remove test files and keys from dependencies
RUN find /usr/lib -type f -path "*/tornado/test/*" -delete && \
    find /usr/lib -type d -path "*/tornado/test" -delete

# SECURITY FIX: nodejs-wheel-binaries (pip package used by Prisma) bundles a complete
# npm with old vulnerable deps at /usr/lib/python3.*/site-packages/nodejs_wheel/.
# Patch every copy of tar, glob, and brace-expansion inside that tree.
RUN GLOBAL="$(npm root -g)" && \
    [ -n "$GLOBAL" ] || { echo "ERROR: npm root -g returned empty; aborting"; exit 1; } && \
    find /usr/lib -type d -name "tar" -path "*/node_modules/tar" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/tar" "$d"; \
    done && \
    find /usr/lib -type d -name "glob" -path "*/node_modules/glob" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/glob" "$d"; \
    done && \
    find /usr/lib -type d -name "brace-expansion" -path "*/node_modules/@isaacs/brace-expansion" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/@isaacs/brace-expansion" "$d"; \
    done && \
    find /usr/lib -type d -name "minimatch" -path "*/node_modules/minimatch" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/minimatch" "$d"; \
    done && \
    find /usr/lib -type d -name "diff" -path "*/node_modules/diff" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/diff" "$d"; \
    done

# Install semantic_router and aurelio-sdk using script
# Convert Windows line endings to Unix and make executable
RUN sed -i 's/\r$//' docker/install_auto_router.sh && chmod +x docker/install_auto_router.sh && ./docker/install_auto_router.sh

# Generate prisma client using the correct schema
RUN prisma generate --schema=./litellm/proxy/schema.prisma

# Copy the built wheel from the builder stage to the runtime stage and install
# it last so Python-only source edits reuse all dependency/runtime setup above.
COPY --from=builder /app/dist/*.whl .
RUN pip install *.whl --no-index --no-deps && rm -f *.whl

# Preserve the historical runtime layout where the repository is available at
# /app (configs, static assets, and source-tree imports from the working
# directory).  Keep this as late as possible: Python-only edits now invalidate
# this cheap copy/chmod tail rather than dependency, Prisma, or CVE-patch layers.
COPY . .
RUN ls -la /app

# Convert Windows line endings to Unix for entrypoint scripts
RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh
RUN sed -i 's/\r$//' docker/prod_entrypoint.sh && chmod +x docker/prod_entrypoint.sh

EXPOSE 4000/tcp

ENTRYPOINT ["docker/prod_entrypoint.sh"]

# Append "--detailed_debug" to the end of CMD to view detailed debug logs
CMD ["--port", "4000"]
