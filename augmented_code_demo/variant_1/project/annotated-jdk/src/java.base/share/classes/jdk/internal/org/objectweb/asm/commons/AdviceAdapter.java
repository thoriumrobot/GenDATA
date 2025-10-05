/*
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
package jdk.internal.org.objectweb.asm.commons;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import jdk.internal.org.objectweb.asm.ConstantDynamic;
    @Positive
import jdk.internal.org.objectweb.asm.Handle;
    @Positive
import jdk.internal.org.objectweb.asm.Label;
    @Positive
import jdk.internal.org.objectweb.asm.MethodVisitor;
    @Positive
import jdk.internal.org.objectweb.asm.Opcodes;
    @Positive
import jdk.internal.org.objectweb.asm.Type;

    @Positive
public abstract class AdviceAdapter extends GeneratorAdapter implements Opcodes {

    @Positive
    protected int methodAccess;

    @Positive
    protected String methodDesc;

    @Positive
    protected AdviceAdapter(final int api, final MethodVisitor methodVisitor, final int access, final String name, final String descriptor) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void visitCode();

    @Positive
    @Override
    @Positive
    public void visitLabel(final Label label);

    @Positive
    @Override
    @Positive
    public void visitInsn(final int opcode);

    @Positive
    @Override
    @Positive
    public void visitVarInsn(final int opcode, final int var);

    @Positive
    @Override
    @Positive
    public void visitFieldInsn(final int opcode, final String owner, final String name, final String descriptor);

    @Positive
    @Override
    @Positive
    public void visitIntInsn(final int opcode, final int operand);

    @Positive
    @Override
    @Positive
    public void visitLdcInsn(final Object value);

    @Positive
    @Override
    @Positive
    public void visitMultiANewArrayInsn(final String descriptor, final int numDimensions);

    @Positive
    @Override
    @Positive
    public void visitTypeInsn(final int opcode, final String type);

    @Positive
    @Override
    @Positive
    public void visitMethodInsn(final int opcodeAndSource, final String owner, final String name, final String descriptor, final boolean isInterface);

    @Positive
    @Override
    @Positive
    public void visitInvokeDynamicInsn(final String name, final String descriptor, final Handle bootstrapMethodHandle, final Object... bootstrapMethodArguments);

    @Positive
    @Override
    @Positive
    public void visitJumpInsn(final int opcode, final Label label);

    @Positive
    @Override
    @Positive
    public void visitLookupSwitchInsn(final Label dflt, final int[] keys, final Label[] labels);

    @Positive
    @Override
    @Positive
    public void visitTableSwitchInsn(final int min, final int max, final Label dflt, final Label... labels);

    @Positive
    @Override
    @Positive
    public void visitTryCatchBlock(final Label start, final Label end, final Label handler, final String type);

    @Positive
    protected void onMethodEnter();

    @Positive
    protected void onMethodExit(final int opcode);
    @Positive
}

// CFWR semantic augmentation - variant 1
