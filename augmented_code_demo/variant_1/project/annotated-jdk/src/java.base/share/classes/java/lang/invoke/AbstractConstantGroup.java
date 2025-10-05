/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.invoke;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.util.*;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import static java.lang.invoke.MethodHandleStatics.rangeCheck1;
    @Positive
import static java.lang.invoke.MethodHandleStatics.rangeCheck2;

    @Positive
abstract class AbstractConstantGroup implements ConstantGroup {

    @Positive
    protected final int size;

    @Positive
    @Override
    @Positive
    public final int size();

    @Positive
    public abstract Object get(int index) throws LinkageError;

    @Positive
    public abstract Object get(int index, Object ifNotPresent);

    @Positive
    public abstract boolean isPresent(int index);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    static class AsIterator implements Iterator<Object> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @Override
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public Object next(@NonEmpty AsIterator this);
    @Positive
    }

    @Positive
    static class SubGroup extends AbstractConstantGroup {

    @Positive
        @Override
    @Positive
        public Object get(int index);

    @Positive
        @Override
    @Positive
        public Object get(int index, Object ifNotPresent);

    @Positive
        @Override
    @Positive
        public boolean isPresent(int index);

    @Positive
        @Override
    @Positive
        public ConstantGroup subGroup(int start, int end);

    @Positive
        @Override
    @Positive
        public List<Object> asList();

    @Positive
        @Override
    @Positive
        public List<Object> asList(Object ifNotPresent);

    @Positive
        @Override
    @Positive
        public int copyConstants(int start, int end, Object[] buf, int pos) throws LinkageError;

    @Positive
        @Override
    @Positive
        public int copyConstants(int start, int end, Object[] buf, int pos, Object ifNotPresent);
    @Positive
    }

    @Positive
    static class AsList extends AbstractList<Object> {

    @Positive
        @Override
    @Positive
        public final int size();

    @Positive
        @Override
    @Positive
        public Object get(int index);

    @Positive
        @Override
    @Positive
        public Iterator<Object> iterator();

    @Positive
        @Override
    @Positive
        public List<Object> subList(int start, int end);

    @Positive
        @Override
    @Positive
        public Object[] toArray();

    @Positive
        @Override
    @Positive
        public <T> T[] toArray(T[] a);
    @Positive
    }

    @Positive
    static abstract class WithCache extends AbstractConstantGroup {

    @Positive
        void initializeCache(List<Object> cacheContents, Object ifNotPresent);

    @Positive
        @Override
    @Positive
        public Object get(int i);

    @Positive
        @Override
    @Positive
        public Object get(int i, Object ifNotAvailable);

    @Positive
        @Override
    @Positive
        public boolean isPresent(int i);

    @Positive
        Object fillCache(int i);

    @Positive
        static Object wrapNull(Object x);

    @Positive
        static Object unwrapNull(Object x);
    @Positive
    }

    @Positive
    static class BSCIWithCache<T> extends WithCache implements BootstrapCallInfo<T> {

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public MethodHandle bootstrapMethod();

    @Positive
        @Override
    @Positive
        public String invocationName();

    @Positive
        @Override
    @Positive
        public T invocationType();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
