/*
 * Copyright (c) 2014, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.annotation.Native;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import java.util.stream.Stream;

@jdk.internal.ValueBased
final class ProcessHandleImpl implements ProcessHandle {

    private static class ExitCompletion extends CompletableFuture<Integer> {
    }

    static CompletableFuture<Integer> completion(long pid, boolean shouldReap);

    @Override
    public CompletableFuture<ProcessHandle> onExit();

    static Optional<ProcessHandle> get(long pid);

    static ProcessHandleImpl getInternal(long pid);

    @Override
    public long pid();

    public static ProcessHandleImpl current();

    public Optional<ProcessHandle> parent();

    boolean destroyProcess(boolean force);

    @Override
    public boolean destroy();

    @Override
    public boolean destroyForcibly();

    @Override
    public boolean supportsNormalTermination();

    @Override
    public boolean isAlive();

    @Override
    public Stream<ProcessHandle> children();

    static Stream<ProcessHandle> children(long pid);

    @Override
    public Stream<ProcessHandle> descendants();

    @Override
    public ProcessHandle.Info info();

    @Override
    public int compareTo(ProcessHandle other);

    @Override
    public String toString();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    static class Info implements ProcessHandle.Info {

        public static ProcessHandle.Info info(long pid, long startTime);

        @Override
        public Optional<String> command();

        @Override
        public Optional<String> commandLine();

        @Override
        public Optional<String[]> arguments();

        @Override
        public Optional<Instant> startInstant();

        @Override
        public Optional<Duration> totalCpuDuration();

        @Override
        public Optional<String> user();

        @Override
        public String toString();
    }
}
