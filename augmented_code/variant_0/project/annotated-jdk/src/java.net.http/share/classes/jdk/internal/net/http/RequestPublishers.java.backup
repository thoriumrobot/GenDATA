/*
 * Copyright (c) 2016, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FilePermission;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.reflect.UndeclaredThrowableException;
import java.net.http.HttpRequest.BodyPublisher;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.AccessControlContext;
import java.security.AccessController;
import java.security.Permission;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Flow;
import java.util.concurrent.Flow.Publisher;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.Supplier;
import jdk.internal.net.http.common.Demand;
import jdk.internal.net.http.common.SequentialScheduler;
import jdk.internal.net.http.common.Utils;

public final class RequestPublishers {

    public static class ByteArrayPublisher implements BodyPublisher {

        public ByteArrayPublisher(byte[] content) {
        }

        public ByteArrayPublisher(byte[] content, int offset, int length) {
        }

        List<ByteBuffer> copy(byte[] content, int offset, int length);

        @Override
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

        @Override
        public long contentLength();
    }

    public static class IterablePublisher implements BodyPublisher {

        public IterablePublisher(Iterable<byte[]> content) {
        }

        class ByteBufferIterator implements Iterator<ByteBuffer> {

            @Override
            @Pure
            public boolean hasNext();

            @Override
            @SideEffectsOnly("this")
            public ByteBuffer next();

            ByteBuffer getBuffer();

            void copy();
        }

        public Iterator<ByteBuffer> iterator();

        @Override
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

        static long computeLength(Iterable<byte[]> bytes);

        @Override
        public long contentLength();
    }

    public static class StringPublisher extends ByteArrayPublisher {

        public StringPublisher(String content, Charset charset) {
        }
    }

    public static class EmptyPublisher implements BodyPublisher {

        @Override
        public long contentLength();

        @Override
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);
    }

    public static class FilePublisher implements BodyPublisher {

        public static FilePublisher create(Path path) throws FileNotFoundException;

        @Override
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

        @Override
        public long contentLength();
    }

    public static class StreamIterator implements Iterator<ByteBuffer> {

        @Override
        @Pure
        public synchronized boolean hasNext();

        @Override
        @SideEffectsOnly("this")
        public synchronized ByteBuffer next();
    }

    public static class InputStreamPublisher implements BodyPublisher {

        public InputStreamPublisher(Supplier<? extends InputStream> streamSupplier) {
        }

        @Override
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

        protected Iterable<ByteBuffer> iterableOf(InputStream is);

        @Override
        public long contentLength();
    }

    public static final class PublisherAdapter implements BodyPublisher {

        public PublisherAdapter(Publisher<? extends ByteBuffer> publisher, long contentLength) {
        }

        @Override
        public final long contentLength();

        @Override
        public final void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);
    }

    public static BodyPublisher concat(BodyPublisher... publishers);

    private static final class AggregatePublisher implements BodyPublisher {

        @Override
        public long contentLength();

        @Override
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);
    }

    private static final class AggregateSubscription implements Flow.Subscription, Flow.Subscriber<ByteBuffer> {

        @Override
        public void request(long n);

        @Override
        public void cancel();

        public void run();

        @Override
        public void onSubscribe(Flow.Subscription subscription);

        @Override
        public void onNext(ByteBuffer item);

        @Override
        public void onError(Throwable throwable);

        @Override
        public void onComplete();
    }
}
