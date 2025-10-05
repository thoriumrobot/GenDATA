/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.jdi;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import com.sun.jdi.AbsentInformationException;
import com.sun.jdi.ClassLoaderReference;
import com.sun.jdi.ClassNotLoadedException;
import com.sun.jdi.ClassObjectReference;
import com.sun.jdi.Field;
import com.sun.jdi.InterfaceType;
import com.sun.jdi.InternalException;
import com.sun.jdi.Location;
import com.sun.jdi.Method;
import com.sun.jdi.ModuleReference;
import com.sun.jdi.ObjectReference;
import com.sun.jdi.ReferenceType;
import com.sun.jdi.Type;
import com.sun.jdi.Value;
import com.sun.jdi.VirtualMachine;

public abstract class ReferenceTypeImpl extends TypeImpl implements ReferenceType {

    protected long ref;

    protected int modifiers;

    protected ReferenceTypeImpl(VirtualMachine aVm, long aRef) {
    }

    void noticeRedefineClass();

    Method getMethodMirror(long ref);

    Field getFieldMirror(long ref);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public int compareTo(ReferenceType object);

    public String signature();

    public String genericSignature();

    public ClassLoaderReference classLoader();

    public ModuleReference module();

    public boolean isPublic();

    public boolean isProtected();

    public boolean isPrivate();

    public boolean isPackagePrivate();

    public boolean isAbstract();

    public boolean isFinal();

    public boolean isStatic();

    public boolean isPrepared();

    public boolean isVerified();

    public boolean isInitialized();

    public boolean failedToInitialize();

    public List<Field> fields();

    abstract List<? extends ReferenceType> inheritedTypes();

    void addVisibleFields(List<Field> visibleList, Map<String, Field> visibleTable, List<String> ambiguousNames);

    public List<Field> visibleFields();

    void addAllFields(List<Field> fieldList, Set<ReferenceType> typeSet);

    public List<Field> allFields();

    public Field fieldByName(String fieldName);

    public List<Method> methods();

    void addToMethodMap(Map<String, Method> methodMap, List<Method> methodList);

    abstract void addVisibleMethods(Map<String, Method> methodMap, Set<InterfaceType> seenInterfaces);

    public List<Method> visibleMethods();

    abstract public List<Method> allMethods();

    public List<Method> methodsByName(String name);

    public List<Method> methodsByName(String name, String signature);

    List<InterfaceType> getInterfaces();

    public List<ReferenceType> nestedTypes();

    public Value getValue(Field sig);

    void validateFieldAccess(Field field);

    void validateFieldSet(Field field);

    public Map<Field, Value> getValues(List<? extends Field> theFields);

    public ClassObjectReference classObject();

    SDE.Stratum stratum(String stratumID);

    public String sourceName() throws AbsentInformationException;

    public List<String> sourceNames(String stratumID) throws AbsentInformationException;

    public List<String> sourcePaths(String stratumID) throws AbsentInformationException;

    String baseSourceName() throws AbsentInformationException;

    String baseSourcePath() throws AbsentInformationException;

    String baseSourceDir();

    public String sourceDebugExtension() throws AbsentInformationException;

    public List<String> availableStrata();

    public String defaultStratum();

    public int modifiers();

    public List<Location> allLineLocations() throws AbsentInformationException;

    public List<Location> allLineLocations(String stratumID, String sourceName) throws AbsentInformationException;

    public List<Location> locationsOfLine(int lineNumber) throws AbsentInformationException;

    public List<Location> locationsOfLine(String stratumID, String sourceName, int lineNumber) throws AbsentInformationException;

    public List<ObjectReference> instances(long maxInstances);

    public int majorVersion();

    public int minorVersion();

    public int constantPoolCount();

    public byte[] constantPool();

    void getModifiers();

    void decodeStatus(int status);

    void updateStatus();

    void markPrepared();

    long ref();

    int indexOf(Method method);

    int indexOf(Field field);

    abstract boolean isAssignableTo(ReferenceType type);

    boolean isAssignableFrom(ReferenceType type);

    boolean isAssignableFrom(ObjectReference object);

    void setStatus(int status);

    void setSignature(String signature);

    void setGenericSignature(String signature);

    Type findType(String signature) throws ClassNotLoadedException;

    String loaderString();
}
