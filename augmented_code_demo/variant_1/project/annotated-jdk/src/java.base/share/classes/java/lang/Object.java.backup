/*
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.guieffect.qual.PolyUI;
import org.checkerframework.checker.guieffect.qual.PolyUIType;
import org.checkerframework.checker.guieffect.qual.SafeEffect;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.mustcall.qual.MustCall;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.checker.tainting.qual.Untainted;
import org.checkerframework.common.aliasing.qual.Unique;
import org.checkerframework.common.reflection.qual.GetClass;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import jdk.internal.vm.annotation.IntrinsicCandidate;

@AnnotatedFor({ "aliasing", "guieffect", "index", "lock", "nullness" })
@PolyUIType
public class Object {

    @Pure
    @IntrinsicCandidate
    @Unique
    @Untainted
    public Object() {
    }

    @GetClass
    @SafeEffect
    @Pure
    @IntrinsicCandidate
    public final native Class<? extends @MustCall() Object> getClass(@PolyUI @GuardSatisfied @UnknownInitialization @UnknownSignedness Object this);

    @Pure
    @IntrinsicCandidate
    public native int hashCode(@GuardSatisfied @UnknownSignedness Object this);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@GuardSatisfied Object this, @GuardSatisfied @Nullable Object obj);

    @SideEffectFree
    @IntrinsicCandidate
    protected native Object clone(@GuardSatisfied Object this) throws CloneNotSupportedException;

    @CFComment({ "nullness: toString() is @SideEffectFree rather than @Pure because it returns a string", "that differs according to ==, and @Deterministic requires that the results of", "two calls of the method are ==." })
    @SideEffectFree
    public String toString(@GuardSatisfied Object this);

    @IntrinsicCandidate
    public final native void notify();

    @IntrinsicCandidate
    public final native void notifyAll();

    public final void wait(@UnknownInitialization Object this) throws InterruptedException;

    public final native void wait(@UnknownInitialization Object this, @NonNegative long timeoutMillis) throws InterruptedException;

    public final void wait(@UnknownInitialization Object this, long timeoutMillis, @NonNegative int nanos) throws InterruptedException;

    @Deprecated()
    protected void finalize() throws Throwable;
}
