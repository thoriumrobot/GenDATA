/*
    @Positive
 * Copyright (c) 2013, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.resources;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.util.AbstractSet;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.concurrent.atomic.AtomicMarkableReference;

    @Positive
public abstract class ParallelListResourceBundle extends ResourceBundle {

    @Positive
    protected ParallelListResourceBundle() {
    @Positive
    }

    @Positive
    protected abstract Object[][] getContents();

    @Positive
    ResourceBundle getParent();

    @Positive
    public void setParallelContents(OpenListResourceBundle rb);

    @Positive
    boolean areParallelContentsComplete();

    @Positive
    @Override
    @Positive
    protected Object handleGetObject(String key);

    @Positive
    @Override
    @Positive
    public Enumeration<String> getKeys();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied @UnknownSignedness String key);

    @Positive
    @Override
    @Positive
    protected Set<String> handleKeySet();

    @Positive
    @Override
    @Positive
    @SuppressWarnings("UnusedAssignment")
    @Positive
    public Set<String> keySet();

    @Positive
    synchronized void resetKeySet();

    @Positive
    void loadLookupTablesIfNecessary();

    @Positive
    private static class KeySet extends AbstractSet<String> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        public Iterator<String> iterator();

    @Positive
        @Override
    @Positive
        public int size();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
