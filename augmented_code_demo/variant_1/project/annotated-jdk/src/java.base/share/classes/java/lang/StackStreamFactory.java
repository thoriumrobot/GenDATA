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
package java.lang;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import jdk.internal.reflect.MethodAccessor;
    @Positive
import jdk.internal.reflect.ConstructorAccessor;
    @Positive
import java.lang.StackWalker.Option;
    @Positive
import java.lang.StackWalker.StackFrame;
    @Positive
import java.lang.annotation.Native;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import static java.lang.StackStreamFactory.WalkerState.*;

    @Positive
final class StackStreamFactory {

    @Positive
    static <T> StackFrameTraverser<T> makeStackTraverser(StackWalker walker, Function<? super Stream<StackFrame>, ? extends T> function);

    @Positive
    static CallerClassFinder makeCallerFinder(StackWalker walker);

    @Positive
    static abstract class AbstractStackWalker<R, T> {

    @Positive
        protected final StackWalker walker;

    @Positive
        protected final Thread thread;

    @Positive
        protected final int maxDepth;

    @Positive
        protected final long mode;

    @Positive
        protected int depth;

    @Positive
        protected FrameBuffer<? extends T> frameBuffer;

    @Positive
        protected long anchor;

    @Positive
        protected AbstractStackWalker(StackWalker walker, int mode) {
    @Positive
        }

    @Positive
        protected AbstractStackWalker(StackWalker walker, int mode, int maxDepth) {
    @Positive
        }

    @Positive
        protected abstract R consumeFrames();

    @Positive
        protected abstract void initFrameBuffer();

    @Positive
        protected abstract int batchSize(int lastBatchFrameCount);

    @Positive
        protected int getNextBatchSize();

    @Positive
        final void checkState(WalkerState state);

    @Positive
        final R walk();

    @Positive
        final Class<?> peekFrame();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        final Class<?> nextFrame();

    @Positive
        @Pure
    @Positive
        final boolean hasNext();
    @Positive
    }

    @Positive
    static class StackFrameTraverser<T> extends AbstractStackWalker<T, StackFrameInfo> implements Spliterator<StackFrame> {

    @Positive
        final class StackFrameBuffer extends FrameBuffer<StackFrameInfo> {

    @Positive
            @Override
    @Positive
            StackFrameInfo[] frames();

    @Positive
            @Override
    @Positive
            void resize(int startIndex, int elements);

    @Positive
            @Override
    @Positive
            StackFrameInfo nextStackFrame();

    @Positive
            @Override
    @Positive
            final Class<?> at(int index);
    @Positive
        }

    @Positive
        StackFrame nextStackFrame();

    @Positive
        @Override
    @Positive
        protected T consumeFrames();

    @Positive
        @Override
    @Positive
        protected void initFrameBuffer();

    @Positive
        @Override
    @Positive
        protected int batchSize(int lastBatchFrameCount);

    @Positive
        @Override
    @Positive
        public Spliterator<StackFrame> trySplit();

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(Consumer<? super StackFrame> action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(Consumer<? super StackFrame> action);
    @Positive
    }

    @Positive
    static final class CallerClassFinder extends AbstractStackWalker<Integer, Class<?>> {

    @Positive
        static final class ClassBuffer extends FrameBuffer<Class<?>> {

    @Positive
            @Override
    @Positive
            Class<?>[] frames();

    @Positive
            @Override
    @Positive
            final Class<?> at(int index);

    @Positive
            @Override
    @Positive
            void resize(int startIndex, int elements);
    @Positive
        }

    @Positive
        Class<?> findCaller();

    @Positive
        @Override
    @Positive
        protected Integer consumeFrames();

    @Positive
        @Override
    @Positive
        protected void initFrameBuffer();

    @Positive
        @Override
    @Positive
        protected int batchSize(int lastBatchFrameCount);

    @Positive
        @Override
    @Positive
        protected int getNextBatchSize();
    @Positive
    }

    @Positive
    static final class LiveStackInfoTraverser<T> extends StackFrameTraverser<T> {

    @Positive
        final class LiveStackFrameBuffer extends FrameBuffer<LiveStackFrameInfo> {

    @Positive
            @Override
    @Positive
            LiveStackFrameInfo[] frames();

    @Positive
            @Override
    @Positive
            void resize(int startIndex, int elements);

    @Positive
            @Override
    @Positive
            LiveStackFrameInfo nextStackFrame();

    @Positive
            @Override
    @Positive
            final Class<?> at(int index);
    @Positive
        }

    @Positive
        @Override
    @Positive
        protected void initFrameBuffer();
    @Positive
    }

    @Positive
    static abstract class FrameBuffer<F> {

    @Positive
        abstract F[] frames();

    @Positive
        abstract void resize(int startIndex, int elements);

    @Positive
        abstract Class<?> at(int index);

    @Positive
        int startIndex();

    @Positive
        F nextStackFrame();

    @Positive
        final int curBatchFrameCount();

    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        final boolean isEmpty();

    @Positive
        final void freeze();

    @Positive
        final boolean isActive();

    @Positive
        final Class<?> next();

    @Positive
        final Class<?> get();

    @Positive
        final int getIndex();

    @Positive
        final void setBatch(int depth, int startIndex, int endIndex);

    @Positive
        final void check(int skipFrames);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
