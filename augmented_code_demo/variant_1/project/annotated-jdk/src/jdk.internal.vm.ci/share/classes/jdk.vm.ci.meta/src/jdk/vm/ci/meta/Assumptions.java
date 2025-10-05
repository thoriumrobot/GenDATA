/*
    @Positive
 * Copyright (c) 2011, 2018, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.
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
package jdk.vm.ci.meta;

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
import java.util.Arrays;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Set;

    @Positive
public final class Assumptions implements Iterable<Assumptions.Assumption> {

    @Positive
    public abstract static class Assumption {
    @Positive
    }

    @Positive
    public static class AssumptionResult<T> {

    @Positive
        public AssumptionResult(T result, Assumption... assumptions) {
    @Positive
        }

    @Positive
        public AssumptionResult(T result) {
    @Positive
        }

    @Positive
        public T getResult();

    @Positive
        public boolean isAssumptionFree();

    @Positive
        public void add(AssumptionResult<T> other);

    @Positive
        public boolean canRecordTo(Assumptions target);

    @Positive
        public void recordTo(Assumptions target);
    @Positive
    }

    @Positive
    public static final class NoFinalizableSubclass extends Assumption {

    @Positive
        public NoFinalizableSubclass(ResolvedJavaType receiverType) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class ConcreteSubtype extends Assumption {

    @Positive
        public final ResolvedJavaType context;

    @Positive
        public final ResolvedJavaType subtype;

    @Positive
        public ConcreteSubtype(ResolvedJavaType context, ResolvedJavaType subtype) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class LeafType extends Assumption {

    @Positive
        public final ResolvedJavaType context;

    @Positive
        public LeafType(ResolvedJavaType context) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class ConcreteMethod extends Assumption {

    @Positive
        public final ResolvedJavaMethod method;

    @Positive
        public final ResolvedJavaType context;

    @Positive
        public final ResolvedJavaMethod impl;

    @Positive
        public ConcreteMethod(ResolvedJavaMethod method, ResolvedJavaType context, ResolvedJavaMethod impl) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class CallSiteTargetValue extends Assumption {

    @Positive
        public final JavaConstant callSite;

    @Positive
        public final JavaConstant methodHandle;

    @Positive
        public CallSiteTargetValue(JavaConstant callSite, JavaConstant methodHandle) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public boolean isEmpty();

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
    @Override
    @Positive
    public Iterator<Assumption> iterator();

    @Positive
    public void recordNoFinalizableSubclassAssumption(ResolvedJavaType receiverType);

    @Positive
    public void recordConcreteSubtype(ResolvedJavaType context, ResolvedJavaType subtype);

    @Positive
    public void recordConcreteMethod(ResolvedJavaMethod method, ResolvedJavaType context, ResolvedJavaMethod impl);

    @Positive
    public void record(Assumption assumption);

    @Positive
    public Assumption[] toArray();

    @Positive
    public void record(Assumptions other);

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
