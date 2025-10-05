/*
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import jdk.internal.misc.TerminatingThreadLocal;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.ref.WeakReference;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

@CFComment({ "nullness: It is permitted to write a subclass that extends ThreadLocal<@NonNull MyType>", "but in such a case:", "* the subclass must override initialValue to return a non-null value", "* the subclass needs to suppress a warning:", "@SuppressWarnings(\"nullness:type.argument\") // initialValue returns non-null" })
@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public class ThreadLocal<@Nullable T> {

    protected T initialValue();

    public static <S> ThreadLocal<S> withInitial(Supplier<? extends S> supplier);

    public ThreadLocal() {
    }

    public T get();

    boolean isPresent();

    public void set(T value);

    public void remove();

    ThreadLocalMap getMap(Thread t);

    void createMap(Thread t, T firstValue);

    static ThreadLocalMap createInheritedMap(ThreadLocalMap parentMap);

    T childValue(T parentValue);

    static final class SuppliedThreadLocal<T> extends ThreadLocal<T> {

        @Override
        protected T initialValue();
    }

    static class ThreadLocalMap {

        static class Entry extends WeakReference<ThreadLocal<?>> {
        }
    }
}
