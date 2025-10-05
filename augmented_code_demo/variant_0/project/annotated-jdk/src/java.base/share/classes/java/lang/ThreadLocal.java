/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import jdk.internal.misc.TerminatingThreadLocal;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.util.function.Supplier;

    @Positive
@CFComment({ "nullness: It is permitted to write a subclass that extends ThreadLocal<@NonNull MyType>", "but in such a case:", "* the subclass must override initialValue to return a non-null value", "* the subclass needs to suppress a warning:", "@SuppressWarnings(\"nullness:type.argument\") // initialValue returns non-null" })
    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class ThreadLocal<@Nullable T> {

    @Positive
    protected T initialValue();

    @Positive
    public static <S> ThreadLocal<S> withInitial(Supplier<? extends S> supplier);

    @Positive
    public ThreadLocal() {
    @Positive
    }

    @Positive
    public T get();

    @Positive
    boolean isPresent();

    @Positive
    public void set(T value);

    @Positive
    public void remove();

    @Positive
    ThreadLocalMap getMap(Thread t);

    @Positive
    void createMap(Thread t, T firstValue);

    @Positive
    static ThreadLocalMap createInheritedMap(ThreadLocalMap parentMap);

    @Positive
    T childValue(T parentValue);

    @Positive
    static final class SuppliedThreadLocal<T> extends ThreadLocal<T> {

    @Positive
        @Override
    @Positive
        protected T initialValue();
    @Positive
    }

    @Positive
    static class ThreadLocalMap {

    @Positive
        static class Entry extends WeakReference<ThreadLocal<?>> {
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
