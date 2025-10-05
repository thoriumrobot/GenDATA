/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import com.sun.jdi.ClassNotLoadedException;
    @Positive
import com.sun.jdi.ClassType;
    @Positive
import com.sun.jdi.Field;
    @Positive
import com.sun.jdi.IncompatibleThreadStateException;
    @Positive
import com.sun.jdi.InterfaceType;
    @Positive
import com.sun.jdi.InternalException;
    @Positive
import com.sun.jdi.InvalidTypeException;
    @Positive
import com.sun.jdi.InvocationException;
    @Positive
import com.sun.jdi.Method;
    @Positive
import com.sun.jdi.ObjectReference;
    @Positive
import com.sun.jdi.ReferenceType;
    @Positive
import com.sun.jdi.ThreadReference;
    @Positive
import com.sun.jdi.Type;
    @Positive
import com.sun.jdi.Value;
    @Positive
import com.sun.jdi.VirtualMachine;

    @Positive
public class ObjectReferenceImpl extends ValueImpl implements ObjectReference, VMListener {

    @Positive
    protected long ref;

    @Positive
    protected static class Cache {
    @Positive
    }

    @Positive
    protected Cache newCache();

    @Positive
    protected Cache getCache();

    @Positive
    protected ClassTypeImpl invokableReferenceType(Method method);

    @Positive
    protected String description();

    @Positive
    public boolean vmSuspended(VMAction action);

    @Positive
    public boolean vmNotSuspended(VMAction action);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public Type type();

    @Positive
    public ReferenceType referenceType();

    @Positive
    public Value getValue(Field sig);

    @Positive
    public Map<Field, Value> getValues(List<? extends Field> theFields);

    @Positive
    public void setValue(Field field, Value value) throws InvalidTypeException, ClassNotLoadedException;

    @Positive
    void validateMethodInvocation(Method method, int options) throws InvalidTypeException, InvocationException;

    @Positive
    void validateClassMethodInvocation(Method method, int options) throws InvalidTypeException, InvocationException;

    @Positive
    void validateIfaceMethodInvocation(Method method, int options) throws InvalidTypeException, InvocationException;

    @Positive
    PacketStream sendInvokeCommand(final ThreadReferenceImpl thread, final ClassTypeImpl refType, final MethodImpl method, final ValueImpl[] args, final int options);

    @Positive
    public Value invokeMethod(ThreadReference threadIntf, Method methodIntf, List<? extends Value> origArguments, int options) throws InvalidTypeException, IncompatibleThreadStateException, InvocationException, ClassNotLoadedException;

    @Positive
    public synchronized void disableCollection();

    @Positive
    public synchronized void enableCollection();

    @Positive
    public boolean isCollected();

    @Positive
    public long uniqueID();

    @Positive
    JDWP.ObjectReference.MonitorInfo jdwpMonitorInfo() throws IncompatibleThreadStateException;

    @Positive
    public List<ThreadReference> waitingThreads() throws IncompatibleThreadStateException;

    @Positive
    public ThreadReference owningThread() throws IncompatibleThreadStateException;

    @Positive
    public int entryCount() throws IncompatibleThreadStateException;

    @Positive
    public List<ObjectReference> referringObjects(long maxReferrers);

    @Positive
    long ref();

    @Positive
    boolean isClassObject();

    @Positive
    ValueImpl prepareForAssignmentTo(ValueContainer destination) throws InvalidTypeException, ClassNotLoadedException;

    @Positive
    void validateAssignment(ValueContainer destination) throws InvalidTypeException, ClassNotLoadedException;

    @Positive
    public String toString();

    @Positive
    byte typeValueKey();
    @Positive
}
