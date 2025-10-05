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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.sun.jdi.ClassNotLoadedException;
import com.sun.jdi.ClassType;
import com.sun.jdi.Field;
import com.sun.jdi.IncompatibleThreadStateException;
import com.sun.jdi.InterfaceType;
import com.sun.jdi.InternalException;
import com.sun.jdi.InvalidTypeException;
import com.sun.jdi.InvocationException;
import com.sun.jdi.Method;
import com.sun.jdi.ObjectReference;
import com.sun.jdi.ReferenceType;
import com.sun.jdi.ThreadReference;
import com.sun.jdi.Type;
import com.sun.jdi.Value;
import com.sun.jdi.VirtualMachine;

public class ObjectReferenceImpl extends ValueImpl implements ObjectReference, VMListener {

    protected long ref;

    protected static class Cache {
    }

    protected Cache newCache();

    protected Cache getCache();

    protected ClassTypeImpl invokableReferenceType(Method method);

    protected String description();

    public boolean vmSuspended(VMAction action);

    public boolean vmNotSuspended(VMAction action);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public Type type();

    public ReferenceType referenceType();

    public Value getValue(Field sig);

    public Map<Field, Value> getValues(List<? extends Field> theFields);

    public void setValue(Field field, Value value) throws InvalidTypeException, ClassNotLoadedException;

    void validateMethodInvocation(Method method, int options) throws InvalidTypeException, InvocationException;

    void validateClassMethodInvocation(Method method, int options) throws InvalidTypeException, InvocationException;

    void validateIfaceMethodInvocation(Method method, int options) throws InvalidTypeException, InvocationException;

    PacketStream sendInvokeCommand(final ThreadReferenceImpl thread, final ClassTypeImpl refType, final MethodImpl method, final ValueImpl[] args, final int options);

    public Value invokeMethod(ThreadReference threadIntf, Method methodIntf, List<? extends Value> origArguments, int options) throws InvalidTypeException, IncompatibleThreadStateException, InvocationException, ClassNotLoadedException;

    public synchronized void disableCollection();

    public synchronized void enableCollection();

    public boolean isCollected();

    public long uniqueID();

    JDWP.ObjectReference.MonitorInfo jdwpMonitorInfo() throws IncompatibleThreadStateException;

    public List<ThreadReference> waitingThreads() throws IncompatibleThreadStateException;

    public ThreadReference owningThread() throws IncompatibleThreadStateException;

    public int entryCount() throws IncompatibleThreadStateException;

    public List<ObjectReference> referringObjects(long maxReferrers);

    long ref();

    boolean isClassObject();

    ValueImpl prepareForAssignmentTo(ValueContainer destination) throws InvalidTypeException, ClassNotLoadedException;

    void validateAssignment(ValueContainer destination) throws InvalidTypeException, ClassNotLoadedException;

    public String toString();

    byte typeValueKey();
}
