/*
    @Positive
 * Copyright (c) 2010, 2013, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import static java.lang.ClassValue.ClassValueMap.probeHomeLocation;
    @Positive
import static java.lang.ClassValue.ClassValueMap.probeBackupLocations;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class ClassValue<T> {

    @Positive
    protected ClassValue() {
    @Positive
    }

    @Positive
    protected abstract T computeValue(Class<?> type);

    @Positive
    public T get(Class<?> type);

    @Positive
    public void remove(Class<?> type);

    @Positive
    void put(Class<?> type, T value);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    Entry<T> castEntry(Entry<?> e);

    @Positive
    boolean match(Entry<?> e);

    @Positive
    static class Identity {
    @Positive
    }

    @Positive
    Version<T> version();

    @Positive
    void bumpVersion();

    @Positive
    static class Version<T> {

    @Positive
        ClassValue<T> classValue();

    @Positive
        Entry<T> promise();

    @Positive
        boolean isLive();
    @Positive
    }

    @Positive
    static class Entry<T> extends WeakReference<Version<T>> {

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        T value();

    @Positive
        boolean isPromise();

    @Positive
        Version<T> version();

    @Positive
        ClassValue<T> classValueOrNull();

    @Positive
        boolean isLive();

    @Positive
        Entry<T> refreshVersion(Version<T> v2);
    @Positive
    }

    @Positive
    static <T> Entry<T> makeEntry(Version<T> explicitVersion, T value);

    @Positive
    static class ClassValueMap extends WeakHashMap<ClassValue.Identity, Entry<?>> {

    @Positive
        Entry<?>[] getCache();

    @Positive
        synchronized <T> Entry<T> startEntry(ClassValue<T> classValue);

    @Positive
        synchronized <T> Entry<T> finishEntry(ClassValue<T> classValue, Entry<T> e);

    @Positive
        synchronized void removeEntry(ClassValue<?> classValue);

    @Positive
        synchronized <T> void changeEntry(ClassValue<T> classValue, T value);

    @Positive
        static Entry<?> loadFromCache(Entry<?>[] cache, int i);

    @Positive
        static <T> Entry<T> probeHomeLocation(Entry<?>[] cache, ClassValue<T> classValue);

    @Positive
        static <T> Entry<T> probeBackupLocations(Entry<?>[] cache, ClassValue<T> classValue);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
