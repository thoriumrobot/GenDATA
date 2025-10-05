/*
    @Positive
 * Copyright (c) 1994, 2016, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;

    @Positive
@CFComment({ "guieffect:", "@PolyUIType class Observable {", "@SafeEffect void addObserver(@PolyUI Observable this, @PolyUI Observer o);", "@SafeEffect void deleteObserver(@PolyUI Observable this, @PolyUI Observer o);", "@PolyUIEffect void notifyObservers(@PolyUI Observable this);", "@PolyUIEffect void notifyObservers(@PolyUI Observable this, Object arg);}" })
    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@Deprecated()
    @Positive
@UsesObjectEquals
    @Positive
public class Observable {

    @Positive
    public Observable() {
    @Positive
    }

    @Positive
    public synchronized void addObserver(@GuardSatisfied Observable this, Observer o);

    @Positive
    public synchronized void deleteObserver(@GuardSatisfied Observable this, @Nullable Observer o);

    @Positive
    public void notifyObservers();

    @Positive
    public void notifyObservers(@Nullable Object arg);

    @Positive
    public synchronized void deleteObservers(@GuardSatisfied Observable this);

    @Positive
    protected synchronized void setChanged();

    @Positive
    protected synchronized void clearChanged();

    @Positive
    public synchronized boolean hasChanged();

    @Positive
    @NonNegative
    @Positive
    public synchronized int countObservers();
    @Positive
}

// CFWR semantic augmentation - variant 1
