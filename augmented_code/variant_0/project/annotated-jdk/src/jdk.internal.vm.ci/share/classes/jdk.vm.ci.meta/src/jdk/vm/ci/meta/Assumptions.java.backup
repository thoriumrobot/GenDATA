/*
 * Copyright (c) 2011, 2018, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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
package jdk.vm.ci.meta;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public final class Assumptions implements Iterable<Assumptions.Assumption> {

    public abstract static class Assumption {
    }

    public static class AssumptionResult<T> {

        public AssumptionResult(T result, Assumption... assumptions) {
        }

        public AssumptionResult(T result) {
        }

        public T getResult();

        public boolean isAssumptionFree();

        public void add(AssumptionResult<T> other);

        public boolean canRecordTo(Assumptions target);

        public void recordTo(Assumptions target);
    }

    public static final class NoFinalizableSubclass extends Assumption {

        public NoFinalizableSubclass(ResolvedJavaType receiverType) {
        }

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);

        @Override
        public String toString();
    }

    public static final class ConcreteSubtype extends Assumption {

        public final ResolvedJavaType context;

        public final ResolvedJavaType subtype;

        public ConcreteSubtype(ResolvedJavaType context, ResolvedJavaType subtype) {
        }

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);

        @Override
        public String toString();
    }

    public static final class LeafType extends Assumption {

        public final ResolvedJavaType context;

        public LeafType(ResolvedJavaType context) {
        }

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);

        @Override
        public String toString();
    }

    public static final class ConcreteMethod extends Assumption {

        public final ResolvedJavaMethod method;

        public final ResolvedJavaType context;

        public final ResolvedJavaMethod impl;

        public ConcreteMethod(ResolvedJavaMethod method, ResolvedJavaType context, ResolvedJavaMethod impl) {
        }

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);

        @Override
        public String toString();
    }

    public static final class CallSiteTargetValue extends Assumption {

        public final JavaConstant callSite;

        public final JavaConstant methodHandle;

        public CallSiteTargetValue(JavaConstant callSite, JavaConstant methodHandle) {
        }

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);

        @Override
        public String toString();
    }

    public boolean isEmpty();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public Iterator<Assumption> iterator();

    public void recordNoFinalizableSubclassAssumption(ResolvedJavaType receiverType);

    public void recordConcreteSubtype(ResolvedJavaType context, ResolvedJavaType subtype);

    public void recordConcreteMethod(ResolvedJavaMethod method, ResolvedJavaType context, ResolvedJavaMethod impl);

    public void record(Assumption assumption);

    public Assumption[] toArray();

    public void record(Assumptions other);

    @Override
    public String toString();
}
