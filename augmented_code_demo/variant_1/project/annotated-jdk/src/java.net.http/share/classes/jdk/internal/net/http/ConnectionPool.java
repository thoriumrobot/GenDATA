/*
    @Positive
 * Copyright (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
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

// CFWR semantic augmentation - variant 1
