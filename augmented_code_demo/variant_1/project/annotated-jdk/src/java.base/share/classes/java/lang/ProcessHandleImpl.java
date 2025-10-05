/*
    @Positive
 * Copyright (c) 2014, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.lang.annotation.Native;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.time.Duration;
    @Positive
import java.time.Instant;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Optional;
    @Positive
import java.util.concurrent.CompletableFuture;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.concurrent.Executor;
    @Positive
import java.util.concurrent.Executors;
    @Positive
import java.util.concurrent.ThreadFactory;
    @Positive
import java.util.concurrent.ThreadLocalRandom;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.Stream;

    @Positive
@jdk.internal.ValueBased
    @Positive
final class ProcessHandleImpl implements ProcessHandle {

    @Positive
    private static class ExitCompletion extends CompletableFuture<Integer> {
    @Positive
    }

    @Positive
    static CompletableFuture<Integer> completion(long pid, boolean shouldReap);

    @Positive
    @Override
    @Positive
    public CompletableFuture<ProcessHandle> onExit();

    @Positive
    static Optional<ProcessHandle> get(long pid);

    @Positive
    static ProcessHandleImpl getInternal(long pid);

    @Positive
    @Override
    @Positive
    public long pid();

    @Positive
    public static ProcessHandleImpl current();

    @Positive
    public Optional<ProcessHandle> parent();

    @Positive
    boolean destroyProcess(boolean force);

    @Positive
    @Override
    @Positive
    public boolean destroy();

    @Positive
    @Override
    @Positive
    public boolean destroyForcibly();

    @Positive
    @Override
    @Positive
    public boolean supportsNormalTermination();

    @Positive
    @Override
    @Positive
    public boolean isAlive();

    @Positive
    @Override
    @Positive
    public Stream<ProcessHandle> children();

    @Positive
    static Stream<ProcessHandle> children(long pid);

    @Positive
    @Override
    @Positive
    public Stream<ProcessHandle> descendants();

    @Positive
    @Override
    @Positive
    public ProcessHandle.Info info();

    @Positive
    @Override
    @Positive
    public int compareTo(ProcessHandle other);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    static class Info implements ProcessHandle.Info {

    @Positive
        public static ProcessHandle.Info info(long pid, long startTime);

    @Positive
        @Override
    @Positive
        public Optional<String> command();

    @Positive
        @Override
    @Positive
        public Optional<String> commandLine();

    @Positive
        @Override
    @Positive
        public Optional<String[]> arguments();

    @Positive
        @Override
    @Positive
        public Optional<Instant> startInstant();

    @Positive
        @Override
    @Positive
        public Optional<Duration> totalCpuDuration();

    @Positive
        @Override
    @Positive
        public Optional<String> user();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
