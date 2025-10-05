/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1998, 2017, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
import java.util.List;
    @Positive
import com.sun.jdi.AbsentInformationException;
    @Positive
import com.sun.jdi.ArrayReference;
    @Positive
import com.sun.jdi.ArrayType;
    @Positive
import com.sun.jdi.ClassNotLoadedException;
    @Positive
import com.sun.jdi.InterfaceType;
    @Positive
import com.sun.jdi.InvalidTypeException;
    @Positive
import com.sun.jdi.Location;
    @Positive
import com.sun.jdi.Method;
    @Positive
import com.sun.jdi.Type;
    @Positive
import com.sun.jdi.Value;
    @Positive
import com.sun.jdi.VirtualMachine;

    @Positive
public abstract class MethodImpl extends TypeComponentImpl implements Method {

    @Positive
    abstract int argSlotCount() throws AbsentInformationException;

    @Positive
    abstract List<Location> allLineLocations(SDE.Stratum stratum, String sourceName) throws AbsentInformationException;

    @Positive
    abstract List<Location> locationsOfLine(SDE.Stratum stratum, String sourceName, int lineNumber) throws AbsentInformationException;

    @Positive
    static MethodImpl createMethodImpl(VirtualMachine vm, ReferenceTypeImpl declaringType, long ref, String name, String signature, String genericSignature, int modifiers);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public final List<Location> allLineLocations() throws AbsentInformationException;

    @Positive
    public List<Location> allLineLocations(String stratumID, String sourceName) throws AbsentInformationException;

    @Positive
    public final List<Location> locationsOfLine(int lineNumber) throws AbsentInformationException;

    @Positive
    public List<Location> locationsOfLine(String stratumID, String sourceName, int lineNumber) throws AbsentInformationException;

    @Positive
    LineInfo codeIndexToLineInfo(SDE.Stratum stratum, long codeIndex);

    @Positive
    public String returnTypeName();

    @Positive
    public Type returnType() throws ClassNotLoadedException;

    @Positive
    public Type findType(String signature) throws ClassNotLoadedException;

    @Positive
    public List<String> argumentTypeNames();

    @Positive
    public List<String> argumentSignatures();

    @Positive
    Type argumentType(int index) throws ClassNotLoadedException;

    @Positive
    public List<Type> argumentTypes() throws ClassNotLoadedException;

    @Positive
    public int compareTo(Method method);

    @Positive
    public boolean isAbstract();

    @Positive
    public boolean isDefault();

    @Positive
    public boolean isSynchronized();

    @Positive
    public boolean isNative();

    @Positive
    public boolean isVarArgs();

    @Positive
    public boolean isBridge();

    @Positive
    public boolean isConstructor();

    @Positive
    public boolean isStaticInitializer();

    @Positive
    public boolean isObsolete();

    @Positive
    class ReturnContainer implements ValueContainer {

    @Positive
        public Type type() throws ClassNotLoadedException;

    @Positive
        public String typeName();

    @Positive
        public String signature();

    @Positive
        public Type findType(String signature) throws ClassNotLoadedException;
    @Positive
    }

    @Positive
    ReturnContainer getReturnValueContainer();

    @Positive
    class ArgumentContainer implements ValueContainer {

    @Positive
        public Type type() throws ClassNotLoadedException;

    @Positive
        public String typeName();

    @Positive
        public String signature();

    @Positive
        public Type findType(String signature) throws ClassNotLoadedException;
    @Positive
    }

    @Positive
    void handleVarArgs(List<Value> arguments) throws ClassNotLoadedException, InvalidTypeException;

    @Positive
    List<Value> validateAndPrepareArgumentsForInvoke(List<? extends Value> origArguments) throws ClassNotLoadedException, InvalidTypeException;

    @Positive
    public String toString();
    @Positive
}
