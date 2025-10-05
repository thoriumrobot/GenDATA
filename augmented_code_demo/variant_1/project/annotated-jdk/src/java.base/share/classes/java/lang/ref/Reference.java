/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.ref;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import jdk.internal.access.JavaLangRefAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.ref.Cleaner;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
@SuppressWarnings({ "rawtypes" })
    @Positive
public abstract class Reference<T> {

    @Positive
    private static class ReferenceHandler extends Thread {

    @Positive
        @SuppressWarnings({ "unchecked" })
    @Positive
        public void run();
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @IntrinsicCandidate
    @Positive
    @Nullable
    @Positive
    public T get(@GuardSatisfied Reference<T> this);

    @Positive
    @Pure
    @Positive
    public final boolean refersTo(T obj);

    @Positive
    @IntrinsicCandidate
    @Positive
    native boolean refersTo0(Object o);

    @Positive
    public void clear();

    @Positive
    T getFromInactiveFinalReference();

    @Positive
    void clearInactiveFinalReference();

    @Positive
    @Deprecated()
    @Positive
    public boolean isEnqueued();

    @Positive
    public boolean enqueue();

    @Positive
    @Override
    @Positive
    protected Object clone() throws CloneNotSupportedException;

    @Positive
    @ForceInline
    @Positive
    @CFComment("nullness: Docs say the parameter can be null, but in practice, calls pass `this`")
    @Positive
    public static void reachabilityFence(Object ref);
    @Positive
}

// CFWR semantic augmentation - variant 1
