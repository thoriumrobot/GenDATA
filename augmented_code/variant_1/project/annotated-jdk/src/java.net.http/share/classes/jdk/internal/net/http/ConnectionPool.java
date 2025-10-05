/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package jdk.internal.net.http;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.System.Logger.Level;
    @Positive
import java.net.InetSocketAddress;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.time.Instant;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.LinkedList;
    @Positive
import java.util.List;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.concurrent.Flow;
    @Positive
import java.util.stream.Collectors;
    @Positive
import jdk.internal.net.http.common.FlowTube;
    @Positive
import jdk.internal.net.http.common.Logger;
    @Positive
import jdk.internal.net.http.common.Utils;

    @Positive
final class ConnectionPool {

    @Positive
    static class CacheKey {

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    final String dbgString();

    @Positive
    synchronized void start();

    @Positive
    static CacheKey cacheKey(InetSocketAddress destination, InetSocketAddress proxy);

    @Positive
    synchronized HttpConnection getConnection(boolean secure, InetSocketAddress addr, InetSocketAddress proxy);

    @Positive
    void returnToPool(HttpConnection conn);

    @Positive
    void returnToPool(HttpConnection conn, Instant now, long keepAlive);

    @Positive
    long purgeExpiredConnectionsAndReturnNextDeadline();

    @Positive
    long purgeExpiredConnectionsAndReturnNextDeadline(Instant now);

    @Positive
    void stop();

    @Positive
    static final class ExpiryEntry {
    @Positive
    }

    @Positive
    private static final class ExpiryList {

    @Positive
        int size();

    @Positive
        boolean purgeMaybeRequired();

    @Positive
        Optional<Instant> nextExpiryDeadline();

    @Positive
        HttpConnection removeOldest();

    @Positive
        void add(HttpConnection conn);

    @Positive
        void add(HttpConnection conn, Instant now, long keepAlive);

    @Positive
        void remove(HttpConnection c);

    @Positive
        List<HttpConnection> purgeUntil(Instant now);

    @Positive
        java.util.stream.Stream<ExpiryEntry> stream();

    @Positive
        void clear();
    @Positive
    }

    @Positive
    @Pure
    @Positive
    synchronized boolean contains(HttpConnection c);

    @Positive
    void cleanup(HttpConnection c, Throwable error);

    @Positive
    private final class CleanupTrigger implements FlowTube.TubeSubscriber, FlowTube.TubePublisher, Flow.Subscription {

    @Positive
        public CleanupTrigger(HttpConnection connection) {
    @Positive
        }

    @Positive
        public boolean isDone();

    @Positive
        @Override
    @Positive
        public void request(long n);

    @Positive
        @Override
    @Positive
        public void cancel();

    @Positive
        @Override
    @Positive
        public void onSubscribe(Flow.Subscription subscription);

    @Positive
        @Override
    @Positive
        public void onError(Throwable error);

    @Positive
        @Override
    @Positive
        public void onComplete();

    @Positive
        @Override
    @Positive
        public void onNext(List<ByteBuffer> item);

    @Positive
        @Override
    @Positive
        public void subscribe(Flow.Subscriber<? super List<ByteBuffer>> subscriber);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
