/*
    @Positive
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.jdi;

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
import java.lang.ref.SoftReference;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import com.sun.jdi.AbsentInformationException;
    @Positive
import com.sun.jdi.ClassLoaderReference;
    @Positive
import com.sun.jdi.ClassNotLoadedException;
    @Positive
import com.sun.jdi.ClassObjectReference;
    @Positive
import com.sun.jdi.Field;
    @Positive
import com.sun.jdi.InterfaceType;
    @Positive
import com.sun.jdi.InternalException;
    @Positive
import com.sun.jdi.Location;
    @Positive
import com.sun.jdi.Method;
    @Positive
import com.sun.jdi.ModuleReference;
    @Positive
import com.sun.jdi.ObjectReference;
    @Positive
import com.sun.jdi.ReferenceType;
    @Positive
import com.sun.jdi.Type;
    @Positive
import com.sun.jdi.Value;
    @Positive
import com.sun.jdi.VirtualMachine;

    @Positive
public abstract class ReferenceTypeImpl extends TypeImpl implements ReferenceType {

    @Positive
    protected long ref;

    @Positive
    protected int modifiers;

    @Positive
    protected ReferenceTypeImpl(VirtualMachine aVm, long aRef) {
    @Positive
    }

    @Positive
    void noticeRedefineClass();

    @Positive
    Method getMethodMirror(long ref);

    @Positive
    Field getFieldMirror(long ref);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public int compareTo(ReferenceType object);

    @Positive
    public String signature();

    @Positive
    public String genericSignature();

    @Positive
    public ClassLoaderReference classLoader();

    @Positive
    public ModuleReference module();

    @Positive
    public boolean isPublic();

    @Positive
    public boolean isProtected();

    @Positive
    public boolean isPrivate();

    @Positive
    public boolean isPackagePrivate();

    @Positive
    public boolean isAbstract();

    @Positive
    public boolean isFinal();

    @Positive
    public boolean isStatic();

    @Positive
    public boolean isPrepared();

    @Positive
    public boolean isVerified();

    @Positive
    public boolean isInitialized();

    @Positive
    public boolean failedToInitialize();

    @Positive
    public List<Field> fields();

    @Positive
    abstract List<? extends ReferenceType> inheritedTypes();

    @Positive
    void addVisibleFields(List<Field> visibleList, Map<String, Field> visibleTable, List<String> ambiguousNames);

    @Positive
    public List<Field> visibleFields();

    @Positive
    void addAllFields(List<Field> fieldList, Set<ReferenceType> typeSet);

    @Positive
    public List<Field> allFields();

    @Positive
    public Field fieldByName(String fieldName);

    @Positive
    public List<Method> methods();

    @Positive
    void addToMethodMap(Map<String, Method> methodMap, List<Method> methodList);

    @Positive
    abstract void addVisibleMethods(Map<String, Method> methodMap, Set<InterfaceType> seenInterfaces);

    @Positive
    public List<Method> visibleMethods();

    @Positive
    abstract public List<Method> allMethods();

    @Positive
    public List<Method> methodsByName(String name);

    @Positive
    public List<Method> methodsByName(String name, String signature);

    @Positive
    List<InterfaceType> getInterfaces();

    @Positive
    public List<ReferenceType> nestedTypes();

    @Positive
    public Value getValue(Field sig);

    @Positive
    void validateFieldAccess(Field field);

    @Positive
    void validateFieldSet(Field field);

    @Positive
    public Map<Field, Value> getValues(List<? extends Field> theFields);

    @Positive
    public ClassObjectReference classObject();

    @Positive
    SDE.Stratum stratum(String stratumID);

    @Positive
    public String sourceName() throws AbsentInformationException;

    @Positive
    public List<String> sourceNames(String stratumID) throws AbsentInformationException;

    @Positive
    public List<String> sourcePaths(String stratumID) throws AbsentInformationException;

    @Positive
    String baseSourceName() throws AbsentInformationException;

    @Positive
    String baseSourcePath() throws AbsentInformationException;

    @Positive
    String baseSourceDir();

    @Positive
    public String sourceDebugExtension() throws AbsentInformationException;

    @Positive
    public List<String> availableStrata();

    @Positive
    public String defaultStratum();

    @Positive
    public int modifiers();

    @Positive
    public List<Location> allLineLocations() throws AbsentInformationException;

    @Positive
    public List<Location> allLineLocations(String stratumID, String sourceName) throws AbsentInformationException;

    @Positive
    public List<Location> locationsOfLine(int lineNumber) throws AbsentInformationException;

    @Positive
    public List<Location> locationsOfLine(String stratumID, String sourceName, int lineNumber) throws AbsentInformationException;

    @Positive
    public List<ObjectReference> instances(long maxInstances);

    @Positive
    public int majorVersion();

    @Positive
    public int minorVersion();

    @Positive
    public int constantPoolCount();

    @Positive
    public byte[] constantPool();

    @Positive
    void getModifiers();

    @Positive
    void decodeStatus(int status);

    @Positive
    void updateStatus();

    @Positive
    void markPrepared();

    @Positive
    long ref();

    @Positive
    int indexOf(Method method);

    @Positive
    int indexOf(Field field);

    @Positive
    abstract boolean isAssignableTo(ReferenceType type);

    @Positive
    boolean isAssignableFrom(ReferenceType type);

    @Positive
    boolean isAssignableFrom(ObjectReference object);

    @Positive
    void setStatus(int status);

    @Positive
    void setSignature(String signature);

    @Positive
    void setGenericSignature(String signature);

    @Positive
    Type findType(String signature) throws ClassNotLoadedException;

    @Positive
    String loaderString();
    @Positive
}

// CFWR semantic augmentation - variant 0
