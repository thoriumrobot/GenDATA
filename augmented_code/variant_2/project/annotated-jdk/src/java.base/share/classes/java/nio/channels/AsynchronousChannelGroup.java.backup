/*
 * Copyright (c) 2007, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.channels.spi.AsynchronousChannelProvider;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class AsynchronousChannelGroup {

    protected AsynchronousChannelGroup(AsynchronousChannelProvider provider) {
    }

    public final AsynchronousChannelProvider provider();

    public static AsynchronousChannelGroup withFixedThreadPool(int nThreads, ThreadFactory threadFactory) throws IOException;

    public static AsynchronousChannelGroup withCachedThreadPool(ExecutorService executor, int initialSize) throws IOException;

    public static AsynchronousChannelGroup withThreadPool(ExecutorService executor) throws IOException;

    public abstract boolean isShutdown();

    public abstract boolean isTerminated();

    public abstract void shutdown();

    public abstract void shutdownNow() throws IOException;

    public abstract boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException;
}
