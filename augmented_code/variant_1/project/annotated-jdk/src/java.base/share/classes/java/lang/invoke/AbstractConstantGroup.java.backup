/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.invoke;

import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.util.*;
import jdk.internal.vm.annotation.Stable;
import static java.lang.invoke.MethodHandleStatics.rangeCheck1;
import static java.lang.invoke.MethodHandleStatics.rangeCheck2;

abstract class AbstractConstantGroup implements ConstantGroup {

    protected final int size;

    @Override
    public final int size();

    public abstract Object get(int index) throws LinkageError;

    public abstract Object get(int index, Object ifNotPresent);

    public abstract boolean isPresent(int index);

    @Override
    public String toString();

    static class AsIterator implements Iterator<Object> {

        @Override
        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean hasNext();

        @Override
        @SideEffectsOnly("this")
        public Object next(@NonEmpty AsIterator this);
    }

    static class SubGroup extends AbstractConstantGroup {

        @Override
        public Object get(int index);

        @Override
        public Object get(int index, Object ifNotPresent);

        @Override
        public boolean isPresent(int index);

        @Override
        public ConstantGroup subGroup(int start, int end);

        @Override
        public List<Object> asList();

        @Override
        public List<Object> asList(Object ifNotPresent);

        @Override
        public int copyConstants(int start, int end, Object[] buf, int pos) throws LinkageError;

        @Override
        public int copyConstants(int start, int end, Object[] buf, int pos, Object ifNotPresent);
    }

    static class AsList extends AbstractList<Object> {

        @Override
        public final int size();

        @Override
        public Object get(int index);

        @Override
        public Iterator<Object> iterator();

        @Override
        public List<Object> subList(int start, int end);

        @Override
        public Object[] toArray();

        @Override
        public <T> T[] toArray(T[] a);
    }

    static abstract class WithCache extends AbstractConstantGroup {

        void initializeCache(List<Object> cacheContents, Object ifNotPresent);

        @Override
        public Object get(int i);

        @Override
        public Object get(int i, Object ifNotAvailable);

        @Override
        public boolean isPresent(int i);

        Object fillCache(int i);

        static Object wrapNull(Object x);

        static Object unwrapNull(Object x);
    }

    static class BSCIWithCache<T> extends WithCache implements BootstrapCallInfo<T> {

        @Override
        public String toString();

        @Override
        public MethodHandle bootstrapMethod();

        @Override
        public String invocationName();

        @Override
        public T invocationType();
    }
}
