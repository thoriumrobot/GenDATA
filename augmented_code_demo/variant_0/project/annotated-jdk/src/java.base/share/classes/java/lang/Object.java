/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.guieffect.qual.PolyUI;
    @Positive
import org.checkerframework.checker.guieffect.qual.PolyUIType;
    @Positive
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCall;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.checker.tainting.qual.Untainted;
    @Positive
import org.checkerframework.common.aliasing.qual.Unique;
    @Positive
import org.checkerframework.common.reflection.qual.GetClass;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "aliasing", "guieffect", "index", "lock", "nullness" })
    @Positive
@PolyUIType
    @Positive
public class Object {

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    @Unique
    @Positive
    @Untainted
    @Positive
    public Object() {
    @Positive
    }

    @Positive
    @GetClass
    @Positive
    @SafeEffect
    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public final native Class<? extends @MustCall() Object> getClass(@PolyUI @GuardSatisfied @UnknownInitialization @UnknownSignedness Object this);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public native int hashCode(@GuardSatisfied @UnknownSignedness Object this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@GuardSatisfied Object this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @SideEffectFree
    @Positive
    @IntrinsicCandidate
    @Positive
    protected native Object clone(@GuardSatisfied Object this) throws CloneNotSupportedException;

    @Positive
    @CFComment({ "nullness: toString() is @SideEffectFree rather than @Pure because it returns a string", "that differs according to ==, and @Deterministic requires that the results of", "two calls of the method are ==." })
    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Object this);

    @Positive
    @IntrinsicCandidate
    @Positive
    public final native void notify();

    @Positive
    @IntrinsicCandidate
    @Positive
    public final native void notifyAll();

    @Positive
    public final void wait(@UnknownInitialization Object this) throws InterruptedException;

    @Positive
    public final native void wait(@UnknownInitialization Object this, @NonNegative long timeoutMillis) throws InterruptedException;

    @Positive
    public final void wait(@UnknownInitialization Object this, long timeoutMillis, @NonNegative int nanos) throws InterruptedException;

    @Positive
    @Deprecated()
    @Positive
    protected void finalize() throws Throwable;
    @Positive
}

// CFWR semantic augmentation - variant 0
