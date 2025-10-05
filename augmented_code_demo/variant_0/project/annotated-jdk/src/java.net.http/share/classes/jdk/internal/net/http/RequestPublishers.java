/*
    @Positive
 * Copyright (c) 2016, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.io.FileInputStream;
    @Positive
import java.io.FileNotFoundException;
    @Positive
import java.io.FilePermission;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.reflect.UndeclaredThrowableException;
    @Positive
import java.net.http.HttpRequest.BodyPublisher;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.Path;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Queue;
    @Positive
import java.util.concurrent.ConcurrentLinkedQueue;
    @Positive
import java.util.concurrent.Flow;
    @Positive
import java.util.concurrent.Flow.Publisher;
    @Positive
import java.util.concurrent.atomic.AtomicReference;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Supplier;
    @Positive
import jdk.internal.net.http.common.Demand;
    @Positive
import jdk.internal.net.http.common.SequentialScheduler;
    @Positive
import jdk.internal.net.http.common.Utils;

    @Positive
public final class RequestPublishers {

    @Positive
    public static class ByteArrayPublisher implements BodyPublisher {

    @Positive
        public ByteArrayPublisher(byte[] content) {
    @Positive
        }

    @Positive
        public ByteArrayPublisher(byte[] content, int offset, int length) {
    @Positive
        }

    @Positive
        List<ByteBuffer> copy(byte[] content, int offset, int length);

    @Positive
        @Override
    @Positive
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

    @Positive
        @Override
    @Positive
        public long contentLength();
    @Positive
    }

    @Positive
    public static class IterablePublisher implements BodyPublisher {

    @Positive
        public IterablePublisher(Iterable<byte[]> content) {
    @Positive
        }

    @Positive
        class ByteBufferIterator implements Iterator<ByteBuffer> {

    @Positive
            @Override
    @Positive
            @Pure
    @Positive
            public boolean hasNext();

    @Positive
            @Override
    @Positive
            @SideEffectsOnly("this")
    @Positive
            public ByteBuffer next();

    @Positive
            ByteBuffer getBuffer();

    @Positive
            void copy();
    @Positive
        }

    @Positive
        public Iterator<ByteBuffer> iterator();

    @Positive
        @Override
    @Positive
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

    @Positive
        static long computeLength(Iterable<byte[]> bytes);

    @Positive
        @Override
    @Positive
        public long contentLength();
    @Positive
    }

    @Positive
    public static class StringPublisher extends ByteArrayPublisher {

    @Positive
        public StringPublisher(String content, Charset charset) {
    @Positive
        }
    @Positive
    }

    @Positive
    public static class EmptyPublisher implements BodyPublisher {

    @Positive
        @Override
    @Positive
        public long contentLength();

    @Positive
        @Override
    @Positive
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);
    @Positive
    }

    @Positive
    public static class FilePublisher implements BodyPublisher {

    @Positive
        public static FilePublisher create(Path path) throws FileNotFoundException;

    @Positive
        @Override
    @Positive
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

    @Positive
        @Override
    @Positive
        public long contentLength();
    @Positive
    }

    @Positive
    public static class StreamIterator implements Iterator<ByteBuffer> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public synchronized boolean hasNext();

    @Positive
        @Override
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public synchronized ByteBuffer next();
    @Positive
    }

    @Positive
    public static class InputStreamPublisher implements BodyPublisher {

    @Positive
        public InputStreamPublisher(Supplier<? extends InputStream> streamSupplier) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);

    @Positive
        protected Iterable<ByteBuffer> iterableOf(InputStream is);

    @Positive
        @Override
    @Positive
        public long contentLength();
    @Positive
    }

    @Positive
    public static final class PublisherAdapter implements BodyPublisher {

    @Positive
        public PublisherAdapter(Publisher<? extends ByteBuffer> publisher, long contentLength) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public final long contentLength();

    @Positive
        @Override
    @Positive
        public final void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);
    @Positive
    }

    @Positive
    public static BodyPublisher concat(BodyPublisher... publishers);

    @Positive
    private static final class AggregatePublisher implements BodyPublisher {

    @Positive
        @Override
    @Positive
        public long contentLength();

    @Positive
        @Override
    @Positive
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber);
    @Positive
    }

    @Positive
    private static final class AggregateSubscription implements Flow.Subscription, Flow.Subscriber<ByteBuffer> {

    @Positive
        @Override
    @Positive
        public void request(long n);

    @Positive
        @Override
    @Positive
        public void cancel();

    @Positive
        public void run();

    @Positive
        @Override
    @Positive
        public void onSubscribe(Flow.Subscription subscription);

    @Positive
        @Override
    @Positive
        public void onNext(ByteBuffer item);

    @Positive
        @Override
    @Positive
        public void onError(Throwable throwable);

    @Positive
        @Override
    @Positive
        public void onComplete();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
