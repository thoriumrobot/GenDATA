/*
 * Copyright (c) 2010, 2013, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.WeakHashMap;
import java.lang.ref.WeakReference;
import java.util.concurrent.atomic.AtomicInteger;
import jdk.internal.misc.Unsafe;
import static java.lang.ClassValue.ClassValueMap.probeHomeLocation;
import static java.lang.ClassValue.ClassValueMap.probeBackupLocations;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class ClassValue<T> {

    protected ClassValue() {
    }

    protected abstract T computeValue(Class<?> type);

    public T get(Class<?> type);

    public void remove(Class<?> type);

    void put(Class<?> type, T value);

    @SuppressWarnings("unchecked")
    Entry<T> castEntry(Entry<?> e);

    boolean match(Entry<?> e);

    static class Identity {
    }

    Version<T> version();

    void bumpVersion();

    static class Version<T> {

        ClassValue<T> classValue();

        Entry<T> promise();

        boolean isLive();
    }

    static class Entry<T> extends WeakReference<Version<T>> {

        @SuppressWarnings("unchecked")
        T value();

        boolean isPromise();

        Version<T> version();

        ClassValue<T> classValueOrNull();

        boolean isLive();

        Entry<T> refreshVersion(Version<T> v2);
    }

    static <T> Entry<T> makeEntry(Version<T> explicitVersion, T value);

    static class ClassValueMap extends WeakHashMap<ClassValue.Identity, Entry<?>> {

        Entry<?>[] getCache();

        synchronized <T> Entry<T> startEntry(ClassValue<T> classValue);

        synchronized <T> Entry<T> finishEntry(ClassValue<T> classValue, Entry<T> e);

        synchronized void removeEntry(ClassValue<?> classValue);

        synchronized <T> void changeEntry(ClassValue<T> classValue, T value);

        static Entry<?> loadFromCache(Entry<?>[] cache, int i);

        static <T> Entry<T> probeHomeLocation(Entry<?>[] cache, ClassValue<T> classValue);

        static <T> Entry<T> probeBackupLocations(Entry<?>[] cache, ClassValue<T> classValue);
    }
}
