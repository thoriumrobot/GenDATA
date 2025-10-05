/*
 * Copyright (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package jdk.internal.net.http;

import org.checkerframework.dataflow.qual.Pure;
import java.io.IOException;
import java.lang.System.Logger.Level;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Flow;
import java.util.stream.Collectors;
import jdk.internal.net.http.common.FlowTube;
import jdk.internal.net.http.common.Logger;
import jdk.internal.net.http.common.Utils;

final class ConnectionPool {

    static class CacheKey {

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();
    }

    final String dbgString();

    synchronized void start();

    static CacheKey cacheKey(InetSocketAddress destination, InetSocketAddress proxy);

    synchronized HttpConnection getConnection(boolean secure, InetSocketAddress addr, InetSocketAddress proxy);

    void returnToPool(HttpConnection conn);

    void returnToPool(HttpConnection conn, Instant now, long keepAlive);

    long purgeExpiredConnectionsAndReturnNextDeadline();

    long purgeExpiredConnectionsAndReturnNextDeadline(Instant now);

    void stop();

    static final class ExpiryEntry {
    }

    private static final class ExpiryList {

        int size();

        boolean purgeMaybeRequired();

        Optional<Instant> nextExpiryDeadline();

        HttpConnection removeOldest();

        void add(HttpConnection conn);

        void add(HttpConnection conn, Instant now, long keepAlive);

        void remove(HttpConnection c);

        List<HttpConnection> purgeUntil(Instant now);

        java.util.stream.Stream<ExpiryEntry> stream();

        void clear();
    }

    @Pure
    synchronized boolean contains(HttpConnection c);

    void cleanup(HttpConnection c, Throwable error);

    private final class CleanupTrigger implements FlowTube.TubeSubscriber, FlowTube.TubePublisher, Flow.Subscription {

        public CleanupTrigger(HttpConnection connection) {
        }

        public boolean isDone();

        @Override
        public void request(long n);

        @Override
        public void cancel();

        @Override
        public void onSubscribe(Flow.Subscription subscription);

        @Override
        public void onError(Throwable error);

        @Override
        public void onComplete();

        @Override
        public void onNext(List<ByteBuffer> item);

        @Override
        public void subscribe(Flow.Subscriber<? super List<ByteBuffer>> subscriber);

        @Override
        public String toString();
    }
}
